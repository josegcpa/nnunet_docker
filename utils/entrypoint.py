import os
import SimpleITK as sitk
import torch
import random
from glob import glob
from utils import (
    resample_image_to_target,
    read_dicom_as_sitk,
    get_study_uid,
    export_dicom_files,
    export_proba_map_and_mask,
    make_parser,
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
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        save_probabilities=True,
    )

    del predictor

    return prediction_files, good_file_paths


if __name__ == "__main__":
    parser = make_parser()

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
    sitk_files, good_file_paths = main(
        model_path=args.model_path.strip(),
        series_paths=series_paths,
        checkpoint_name=args.checkpoint_name.strip(),
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        is_dicom=args.is_dicom,
        use_folds=folds,
        use_mirroring=args.tta,
        study_name=args.study_uid,
    )

    suffix = args.suffix
    output_names = {
        "prediction": (
            "prediction" if suffix is None else f"prediction_{suffix}"
        ),
        "probabilities": (
            "probabilities" if suffix is None else f"proba_{suffix}"
        ),
        "struct": "struct" if suffix is None else f"struct_{suffix}",
    }

    proba_map, mask, empty = export_proba_map_and_mask(
        sitk_files,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        proba_threshold=args.proba_threshold,
        intersect_with=args.intersect_with,
        min_intersection=args.min_intersection,
        output_proba_map_file_name=output_names["probabilities"],
        output_mask_file_name=output_names["prediction"],
        class_idx=args.class_idx,
    )

    if args.save_nifti_inputs is True:
        for sitk_file in sitk_files:
            basename = os.path.basename(sitk_file)
            for s in [".mha", ".nii.gz"]:
                basename = basename.rstrip(s)
            output_nifti = f"{args.output_dir}/{basename}.nii.gz"
            print(f"Copying Nifti to {output_nifti}")
            sitk.WriteImage(sitk.ReadImage(sitk_file), output_nifti)

    if (empty is True) and (args.empty_segment_metadata is not None):
        metadata_path = args.empty_segment_metadata
        fractional_metadata_path = args.empty_segment_metadata
        save_dicom = True
    elif empty is False:
        metadata_path = args.metadata_path
        fractional_metadata_path = args.fractional_metadata_path
        save_dicom = True
    else:
        print("Mask is empty, skipping DICOM formats")

    if save_dicom is True and args.is_dicom is True:
        export_dicom_files(
            output_dir=args.output_dir,
            prediction_name=output_names["prediction"],
            probabilities_name=output_names["probabilities"],
            struct_name=output_names["struct"],
            metadata_path=metadata_path,
            fractional_metadata_path=fractional_metadata_path,
            fractional_as_segments=args.fractional_as_segments,
            dicom_file_paths=good_file_paths,
            mask=mask,
            proba_map=proba_map,
            save_proba_map=args.proba_map,
            save_rt_struct=args.rt_struct_output,
            class_idx=args.class_idx,
        )
