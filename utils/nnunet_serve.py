import time
import os
import subprocess as sp
import json
import yaml
import fastapi
import SimpleITK as sitk
import torch
from fastapi import HTTPException
from glob import glob
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from utils import (
    resample_image_to_target,
    read_dicom_as_sitk,
    export_to_dicom_seg,
    export_to_dicom_struct,
    export_proba_map,
    export_fractional_dicom_seg,
)

os.environ["nnUNet_preprocessed"] = "tmp/preproc"
os.environ["nnUNet_raw"] = "tmp"
os.environ["nnUNet_results"] = "tmp"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


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
    series_paths: list[str] | list[list[str]] = Field(
        description="Paths or list of paths to series."
    )
    output_dir: str = Field("Output directory.")
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
    use_folds: list[int] = Field(
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


def inference(
    predictor: nnUNetPredictor,
    nnunet_path: str,
    series_paths: list[str],
    output_dir: str,
    prediction_idx: int | list[int] = 1,
    checkpoint_name: str = "checkpoint_best.pth",
    tmp_dir: str = ".tmp",
    is_dicom: bool = False,
    use_folds: tuple[int] = (0,),
    proba_map: bool = False,
    proba_threshold: float = 0.1,
    min_confidence: float = None,
    intersect_with: str | sitk.Image = None,
    min_overlap: float = 0.1,
):

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

    proba_map = export_proba_map(
        prediction_files,
        output_dir=output_dir,
        min_confidence=min_confidence,
        proba_threshold=proba_threshold,
        intersect_with=intersect_with,
        min_overlap=min_overlap,
        class_idx=prediction_idx,
    )

    output_mask_path = f"{output_dir}/prediction.nii.gz"
    mask = sitk.Cast(proba_map > 0.5, sitk.sitkUInt16)
    sitk.WriteImage(mask, output_mask_path)

    return (
        prediction_files,
        output_mask_path,
        good_file_paths,
        proba_map,
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
    proba_threshold: float = 0.1,
    min_confidence: float | None = None,
    intersect_with: str | sitk.Image | None = None,
    min_overlap: float = 0.1,
    save_proba_map: bool = False,
    save_nifti_inputs: bool = False,
    save_rt_struct_output: bool = False,
    suffix: str | None = None,
    metadata_path: str | None = None,
):
    def coherce_to_list(obj, n: int):
        if isinstance(obj, (list, tuple)):
            if len(obj) != n:
                raise ValueError(f"{obj} should have length {n}")
        else:
            obj = [obj for obj in range(n)]
        return obj

    output_dir = output_dir.strip().rstrip("/")
    use_folds = [int(f) for f in use_folds]

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
        min_confidence = coherce_to_list(proba_threshold, len(nnunet_path))

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
                proba_map=save_proba_map,
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
            proba_map=save_proba_map,
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
        if is_dicom is True:
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


app = fastapi.FastAPI()

with open("model-serve-spec.yaml") as o:
    models_specs = yaml.safe_load(o)
if "model_folder" not in models_specs:
    raise ValueError("model_folder must be specified in model-serve-spec.yaml")
model_folder = models_specs["model_folder"]
model_paths = [os.path.dirname(x) for x in Path(model_folder).rglob("fold_0")]
model_dictionary = {
    m.split(os.sep)[-2]: {
        "path": m,
        "model_information": get_info(f"{m}/dataset.json"),
    }
    for m in model_paths
}
model_dictionary = {
    m: {**model_dictionary[m], **models_specs["models"].get(m, None)}
    for m in model_dictionary
    if m in models_specs["models"]
}


@app.get("/model_info")
def model_info():
    return model_dictionary


@app.post("/infer")
def infer(inference_request: InferenceRequest):
    a = time.time()
    free = False
    while free is False:
        gpu_memory = get_gpu_memory()
        max_gpu_memory = max(gpu_memory)
        device_id = [
            i
            for i in range(len(gpu_memory))
            if gpu_memory[i] == max_gpu_memory
        ][0]
        if max_gpu_memory > 4000:
            free = True

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cuda", device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    prior_use_mirroring = predictor.use_mirroring
    nnunet_id = inference_request.nnunet_id
    if isinstance(nnunet_id, str):
        if nnunet_id not in model_dictionary:
            raise HTTPException(
                404, f"{nnunet_id} is not a valid nnUNet model"
            )
        nnunet_info = model_dictionary[nnunet_id]
        nnunet_path, metadata_path = nnunet_info["path"], nnunet_info.get(
            "metadata", None
        )
    else:
        nnunet_path = []
        for nn in nnunet_id:
            nnunet_info = model_dictionary[nn]
            nnunet_path.append(nnunet_info["path"])
        metadata_path = nnunet_info.get("metadata", None)

    params = inference_request.__dict__
    if "tta" in inference_request:
        predictor.use_mirroring = inference_request.tta
    del params["nnunet_id"]
    del params["tta"]
    output_paths = wraper(
        **params,
        predictor=predictor,
        nnunet_path=nnunet_path,
        metadata_path=metadata_path,
    )
    predictor.use_mirroring = prior_use_mirroring
    b = time.time()

    del predictor
    torch.cuda.empty_cache()

    return {
        "time_elapsed": b - a,
        "gpu": device_id,
        "nnunet_path": nnunet_path,
        "metadata_path": metadata_path,
        **output_paths,
    }
