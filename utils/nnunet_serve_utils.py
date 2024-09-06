import os
import subprocess as sp
import json
import SimpleITK as sitk
import torch
from glob import glob
from pydantic import BaseModel, ConfigDict, Field
from utils import (
    Folds,
    resample_image_to_target,
    read_dicom_as_sitk,
    export_to_dicom_seg,
    export_to_dicom_struct,
    export_proba_map,
    export_fractional_dicom_seg,
)
from typing import Any

SUCCESS_STATUS = "done"
FAILURE_STATUS = "failed"

os.environ["nnUNet_preprocessed"] = "tmp/preproc"
os.environ["nnUNet_raw"] = "tmp"
os.environ["nnUNet_results"] = "tmp"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa


class InferenceRequest(BaseModel):
    """
    Data model for the inference request from local data. Supports providing
    multiple nnUNet model identifiers (``nnunet_id``) which in turn allows for
    intersection-based filtering of downstream results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nnunet_id: str | list[str] = Field(
        description="nnUnet model identifier or list of nnUNet model identifiers."
    )
    study_path: str = Field(
        description="Path to study folder or list of paths to studies."
    )
    series_folders: list[str] | list[list[str]] = Field(
        description="Series folder names or list of series folder names (relative to study_path).",
        default=None,
    )
    output_dir: str = Field(description="Output directory.")
    prediction_idx: int | list[int] | list[list[int] | int] = Field(
        description="Prediction index or indices which are kept after each prediction",
        default=1,
    )
    checkpoint_name: str = Field(
        description="nnUNet checkpoint name", default="checkpoint_best.pth"
    )
    tmp_dir: str = Field(
        description="Directory for temporary outputs", default=".tmp"
    )
    is_dicom: bool = Field(
        description="Whether series_paths refers to DICOM series folders",
        default=False,
    )
    tta: bool = Field(
        description="Whether to apply test-time augmentation (use_mirroring)",
        default=True,
    )
    use_folds: Folds = Field(
        description="Which folds should be used", default_factory=lambda: [0]
    )
    proba_threshold: float | list[float] = Field(
        description="Probability threshold for model output", default=0.1
    )
    min_confidence: float | list[float] | None = Field(
        description="Minimum confidence for model output", default=None
    )
    intersect_with: str | sitk.Image | None = Field(
        description="Intersects output with this mask and if relative \
            intersection < min_overlap this is set to 0",
        default=None,
    )
    min_overlap: float = Field(
        description="Minimum overlap for intersection", default=0.1
    )
    save_proba_map: bool = Field(
        description="Saves the probability map", default=False
    )
    save_nifti_inputs: bool = Field(
        description="Saves the Nifti inputs in the output folder if input is DICOM",
        default=False,
    )
    save_rt_struct_output: bool = Field(
        description="Saves the output as an RT struct file", default=False
    )
    suffix: str | None = Field(
        description="Suffix for predictions", default=None
    )


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return memory_free_values


def get_series_paths(
    study_path: str,
    series_folders: list[str] | list[list[str]] | None,
    n: int | None,
) -> tuple[list[str], str, str] | tuple[list[list[str]], str, str]:
    print(series_folders)
    if series_folders is None:
        return (
            None,
            FAILURE_STATUS,
            "series_folders must be defined",
        )
    if n is None:
        series_paths = [os.path.join(study_path, x) for x in series_folders]
    else:
        study_path = [study_path for _ in range(n)]
        series_paths = []
        if n != len(series_folders):
            return (
                None,
                FAILURE_STATUS,
                "series_folders and nnunet_id must be the same length",
            )
        for i in range(len(study_path)):
            series_paths.append(
                [os.path.join(study_path[i], x) for x in series_folders[i]]
            )

    return series_paths, SUCCESS_STATUS, None


def wait_for_gpu(min_mem: int) -> int:
    free = False
    while free is False:
        gpu_memory = get_gpu_memory()
        max_gpu_memory = max(gpu_memory)
        device_id = [
            i for i in range(len(gpu_memory)) if gpu_memory[i] == max_gpu_memory
        ][0]
        if max_gpu_memory > min_mem:
            free = True
    return device_id


def get_default_params(default_args: dict | list[dict]) -> dict:
    args_with_mult_support = [
        "proba_threshold",
        "min_confidence",
        "prediction_idx",
        "series_folders",
    ]
    if isinstance(default_args, dict):
        default_params = default_args
    else:
        default_params = {}
        for curr_default_args in default_args:
            for k in curr_default_args:
                if k in args_with_mult_support:
                    if k not in default_params:
                        default_params[k] = []
                    default_params[k].append(curr_default_args[k])
                else:
                    default_params[k] = curr_default_args[k]
    return default_params


def predict(
    series_paths: list,
    metadata_path: str,
    mirroring: bool,
    device_id: int,
    params: dict,
    nnunet_path: str,
):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=mirroring,
        device=torch.device("cuda", device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    for k in [
        "nnunet_id",
        "tta",
        "min_mem",
        "aliases",
        "study_path",
        "series_folders",
    ]:
        if k in params:
            del params[k]

    output_paths = wraper(
        **params,
        series_paths=series_paths,
        predictor=predictor,
        nnunet_path=nnunet_path,
        metadata_path=metadata_path,
    )
    del predictor
    torch.cuda.empty_cache()
    return output_paths


def inference(
    predictor: nnUNetPredictor,
    nnunet_path: str,
    series_paths: list[str],
    output_dir: str,
    prediction_idx: int | list[int] = 1,
    checkpoint_name: str = "checkpoint_best.pth",
    tmp_dir: str = ".tmp",
    is_dicom: bool = False,
    use_folds: Folds = (0,),
    proba_threshold: float = 0.1,
    min_confidence: float | None = None,
    intersect_with: str | sitk.Image | None = None,
    min_overlap: float = 0.1,
) -> tuple[list[str], str, list[list[str]], sitk.Image]:

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        nnunet_path,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
    )

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    prediction_files = []
    term = predictor.dataset_json["file_ending"]
    if len(predictor.dataset_json["channel_names"]) != len(series_paths):
        exp_chan = predictor.dataset_json["channel_names"]
        raise ValueError(
            f"series_paths should have length {len(exp_chan)} ({exp_chan}) but has length {len(series_paths)}"
        )
    if is_dicom is True:
        sitk_images = [
            read_dicom_as_sitk(glob(f"{series_path}/*dcm"))
            for series_path in series_paths
        ]
        good_file_paths = [x[1] for x in sitk_images]
        sitk_images = [x[0] for x in sitk_images]
        for idx in range(1, len(sitk_images)):
            sitk_images[idx] = resample_image_to_target(
                sitk_images[idx], target=sitk_images[0]
            )
        for idx, image in enumerate(sitk_images):
            file_termination = f"{idx}".rjust(4, "0")
            sitk_image_path = f"{tmp_dir}/volume_{file_termination}{term}"
            prediction_files.append(sitk_image_path)
            sitk.WriteImage(image, sitk_image_path)
    else:
        for idx, series_path in enumerate(series_paths):
            file_termination = f"{idx}".rjust(4, "0")
            tmp_path = f"{tmp_dir}/volume_{file_termination}{term}"
            # rewrite with correct nnunet formatting
            sitk.WriteImage(sitk.ReadImage(series_path), tmp_path)
            prediction_files.append(tmp_path)

    predictor.predict_from_files(
        [prediction_files],
        output_dir,
        save_probabilities=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    probability_map = export_proba_map(
        prediction_files,
        output_dir=output_dir,
        min_confidence=min_confidence,
        proba_threshold=proba_threshold,
        intersect_with=intersect_with,
        min_intersection=min_overlap,
        class_idx=prediction_idx,
    )

    output_mask_path = f"{output_dir}/prediction.nii.gz"
    mask = sitk.Cast(probability_map > 0.5, sitk.sitkUInt16)
    sitk.WriteImage(mask, output_mask_path)

    return (
        prediction_files,
        output_mask_path,
        good_file_paths,
        probability_map,
    )


def get_info(dataset_json_path: str) -> dict:
    with open(dataset_json_path) as o:
        return json.load(o)


def wraper(
    predictor: nnUNetPredictor,
    nnunet_path: str | list[str],
    series_paths: list[str] | list[list[str]],
    output_dir: str,
    prediction_idx: int | list[int] | list[list[int]] = 1,
    checkpoint_name: str = "checkpoint_best.pth",
    tmp_dir: str = ".tmp",
    is_dicom: bool = False,
    use_folds: tuple[int] = (0,),
    proba_threshold: float | tuple[float] | list[float] = 0.1,
    min_confidence: float | tuple[float] | list[float] | None = None,
    intersect_with: str | sitk.Image | None = None,
    min_overlap: float = 0.1,
    save_proba_map: bool = False,
    save_nifti_inputs: bool = False,
    save_rt_struct_output: bool = False,
    suffix: str | None = None,
    metadata_path: str | None = None,
):
    def coherce_to_list(obj: Any, n: int) -> list[Any] | tuple[Any]:
        if isinstance(obj, (list, tuple)):
            if len(obj) != n:
                raise ValueError(f"{obj} should have length {n}")
        else:
            obj = [obj for obj in range(n)]
        return obj

    output_dir = output_dir.strip().rstrip("/")

    if isinstance(series_paths, (tuple, list)) is False:
        raise ValueError(
            f"series_paths should be list of strings or list of list of strings (is {series_paths})"
        )
    if isinstance(nnunet_path, (list, tuple)):
        # minimal input parsing
        series_paths_list = None
        prediction_idx_list = None
        if isinstance(series_paths, (tuple, list)):
            if isinstance(series_paths[0], (list, tuple)):
                series_paths_list = series_paths
            elif isinstance(series_paths[0], str):
                series_paths_list = [series_paths for _ in nnunet_path]
        if isinstance(prediction_idx, int):
            prediction_idx_list = [prediction_idx for _ in nnunet_path]
        elif isinstance(prediction_idx, (tuple, list)):
            if isinstance(prediction_idx[0], (list, tuple)):
                prediction_idx_list = prediction_idx
            elif isinstance(prediction_idx[0], int):
                prediction_idx_list = [prediction_idx for _ in nnunet_path]
        proba_threshold = coherce_to_list(proba_threshold, len(nnunet_path))
        min_confidence = coherce_to_list(min_confidence, len(nnunet_path))

        if series_paths_list is None:
            raise ValueError(
                f"series_paths should be list of strings or list of list of strings (is {series_paths})"
            )
        for i in range(len(nnunet_path)):
            if i == (len(nnunet_path) - 1):
                out = output_dir
            else:
                out = tmp_dir
            sitk_files, mask_path, good_file_paths, proba_map = inference(
                predictor=predictor,
                nnunet_path=nnunet_path[i].strip(),
                series_paths=series_paths[i],
                prediction_idx=prediction_idx_list[i],
                checkpoint_name=checkpoint_name.strip(),
                output_dir=out,
                tmp_dir=tmp_dir,
                is_dicom=is_dicom,
                use_folds=use_folds,
                proba_threshold=proba_threshold[i],
                min_confidence=min_confidence[i],
                intersect_with=intersect_with,
                min_overlap=min_overlap,
            )
            intersect_with = mask_path
    else:
        sitk_files, mask_path, good_file_paths, proba_map = inference(
            predictor=predictor,
            nnunet_path=nnunet_path.strip(),
            series_paths=series_paths,
            checkpoint_name=checkpoint_name.strip(),
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            is_dicom=is_dicom,
            use_folds=use_folds,
            proba_threshold=proba_threshold,
            min_confidence=min_confidence,
            intersect_with=intersect_with,
            min_overlap=min_overlap,
        )

    mask = sitk.ReadImage(mask_path)

    output_names = {
        "prediction": (
            "prediction" if suffix is None else f"prediction_{suffix}"
        ),
        "probabilities": (
            "probabilities" if suffix is None else f"proba_{suffix}"
        ),
        "struct": "struct" if suffix is None else f"struct_{suffix}",
    }

    output_paths = {
        "nifti_prediction": mask_path,
        "nifti_probabilities": f"{output_dir}/{output_names['probabilities']}.nii.gz",
    }

    if save_nifti_inputs is True:
        niftis = []
        for sitk_file in sitk_files:
            basename = os.path.basename(sitk_file)
            for s in [".mha", ".nii.gz"]:
                basename = basename.rstrip(s)
            output_nifti = f"{output_dir}/{basename}.nii.gz"
            print(f"Copying Nifti to {output_nifti}")
            sitk.WriteImage(sitk.ReadImage(sitk_file), output_nifti)
            niftis.append(output_nifti)
        output_paths["nifti_inputs"] = niftis

    if is_dicom is True:
        if metadata_path is None:
            raise ValueError(
                "if is_dicom is True metadata_path must be specified"
            )
        status = export_to_dicom_seg(
            mask,
            metadata_path=metadata_path,
            file_paths=good_file_paths,
            output_dir=output_dir,
            output_file_name=output_names["prediction"],
        )
        if "empty" in status:
            print("Mask is empty, skipping DICOMseg/RTstruct")
        elif save_rt_struct_output:
            export_to_dicom_struct(
                mask,
                metadata_path=metadata_path,
                file_paths=good_file_paths,
                output_dir=output_dir,
                output_file_name=output_names["struct"],
            )
            output_paths["dicom_struct"] = (
                f"{output_dir}/{output_names['struct']}.dcm"
            )
            output_paths["dicom_segmentation"] = (
                f"{output_dir}/{output_names['prediction']}.dcm"
            )
        else:
            output_paths["dicom_segmentation"] = (
                f"{output_dir}/{output_names['prediction']}.dcm"
            )

        if save_proba_map is True:
            export_fractional_dicom_seg(
                proba_map,
                metadata_path=metadata_path,
                file_paths=good_file_paths,
                output_dir=output_dir,
                output_file_name=output_names["probabilities"],
            )
            output_paths["dicom_fractional_segmentation"] = (
                f"{output_dir}/{output_names['probabilities']}.dcm"
            )

    return output_paths
