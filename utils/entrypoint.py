import os
import SimpleITK as sitk
import torch
import random
import json
import numpy as np
from glob import glob
from utils import (
    resample_image_to_target,
    read_dicom_as_sitk,
    get_study_uid,
    export_to_dicom_seg,
    export_to_dicom_struct,
    export_proba_map,
    export_fractional_dicom_seg,
)


def main(
    model_path: str,
    series_paths: list[str],
    checkpoint_name: str,
    output_dir: str,
    tmp_dir: str = ".tmp",
    is_dicom: bool = False,
    use_mirroring: bool = True,
    use_folds: tuple[int] = (0,),
    study_name: str = None,
    proba_map: bool = False,
):

    os.environ["nnUNet_preprocessed"] = "tmp/preproc"
    os.environ["nnUNet_raw"] = series_paths[0]
    os.environ["nnUNet_results"] = model_path

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
    )

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    prediction_files = []
    term = predictor.dataset_json["file_ending"]
    if is_dicom is True:
        study_name = get_study_uid(series_paths[0])
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
        if study_name is None:
            # creates random study number if necessary
            study_name = str(random.randint(0, 1e6))
        for idx, series_path in enumerate(series_paths):
            file_termination = f"{idx}".rjust(4, "0")
            tmp_path = f"{tmp_dir}/volume_{file_termination}{term}"
            # rewrite with correct nnunet formatting
            sitk.WriteImage(sitk.ReadImage(series_path), tmp_path)
            prediction_files.append(tmp_path)

    predictor.predict_from_files(
        [prediction_files],
        output_dir,
        save_probabilities=proba_map,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    mask_path = f"{output_dir}/volume{term}"
    output_mask_path = f"{output_dir}/prediction.nii.gz"
    sitk.WriteImage(sitk.ReadImage(mask_path), output_mask_path)

    return prediction_files, output_mask_path, study_name, good_file_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Entrypoint for nnUNet prediction. Handles all data format conversions."
    )

    parser.add_argument(
        "--series_paths",
        "-i",
        nargs="+",
        help="Path to input series",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        "-m",
        help="Path to nnUNet model folder",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_name",
        "-ckpt",
        help="Checkpoint name for nnUNet",
        default="checkpoint_best.pth",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to output directory",
        required=True,
    )
    parser.add_argument(
        "--metadata_path",
        "-M",
        help="Path to metadata template for DICOM-Seg output",
        required=True,
    )
    parser.add_argument(
        "--study_uid",
        "-s",
        help="Study UID if series are SimpleITK-readable files",
        default=None,
    )
    parser.add_argument(
        "--folds",
        "-f",
        help="Sets which folds should be used with nnUNet",
        nargs="+",
        type=str,
        default=(0,),
    )
    parser.add_argument(
        "--tta",
        "-t",
        help="Uses test-time augmentation during prediction",
        action="store_true",
    )
    parser.add_argument(
        "--tmp_dir",
        help="Temporary directory",
        default=".tmp",
    )
    parser.add_argument(
        "--is_dicom",
        "-D",
        help="Assumes input is DICOM (and also converts to DICOM seg; \
            prediction.dcm in output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--proba_map",
        "-p",
        help="Produces a Nifti format probability map (probabilities.nii.gz \
            in output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--rt_struct_output",
        help="Produces a DICOM RT Struct file (struct.dcm in output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--save_nifti_inputs",
        "-S",
        help="Moves Nifti inputs to output folder (volume_XXXX.nii.gz in \
            output_dir)",
        action="store_true",
    )

    args = parser.parse_args()

    # easier to adapt to docker
    if "SERIES_PATHS" in os.environ:
        series_paths = os.environ["SERIES_PATHS"].split(" ")
    else:
        series_paths = args.series_paths
    series_paths = [s.strip() for s in series_paths]

    args.output_dir = args.output_dir.strip().rstrip("/")
    folds = []
    for f in args.folds:
        folds.append(int(f))
    sitk_files, mask_path, study_name, good_file_paths = main(
        model_path=args.model_path.strip(),
        series_paths=series_paths,
        checkpoint_name=args.checkpoint_name.strip(),
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        is_dicom=args.is_dicom,
        use_folds=folds,
        use_mirroring=args.tta,
        study_name=args.study_uid,
        proba_map=args.proba_map,
    )

    mask = sitk.ReadImage(mask_path)

    if args.save_nifti_inputs is True:
        for sitk_file in sitk_files:
            basename = os.path.basename(sitk_file)
            for s in [".mha", ".nii.gz"]:
                basename = basename.rstrip(s)
            output_nifti = f"{args.output_dir}/{basename}.nii.gz"
            print(f"Copying Nifti to {output_nifti}")
            sitk.WriteImage(sitk.ReadImage(sitk_file), output_nifti)

    if args.is_dicom is True:
        status = export_to_dicom_seg(
            mask,
            metadata_path=args.metadata_path,
            file_paths=good_file_paths,
            output_dir=args.output_dir,
        )
        if "empty" in status:
            print("Mask is empty, skipping DICOMseg/RTstruct")
            exit()

    if args.rt_struct_output:
        export_to_dicom_struct(
            mask,
            metadata_path=args.metadata_path,
            file_paths=good_file_paths,
            output_dir=args.output_dir,
        )

    if args.proba_map is True:
        proba_map = export_proba_map(sitk_files, args.output_dir)

        if args.is_dicom is True:
            export_fractional_dicom_seg(
                proba_map,
                metadata_path=args.metadata_path,
                file_paths=good_file_paths,
                output_dir=args.output_dir,
            )
