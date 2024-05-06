import os
import SimpleITK as sitk
import torch
import random
import json
import numpy as np
from glob import glob
from utils import resample_image_to_target, read_dicom_as_sitk, get_study_uid


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
        perform_everything_on_gpu=True,
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
        [prediction_files], output_dir, save_probabilities=proba_map
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
    args.series_paths = os.environ.get("SERIES_PATHS", args.series_paths)

    args.output_dir = args.output_dir.strip().rstrip("/")
    folds = []
    for f in args.folds:
        if "," in f:
            folds.extend([int(x) for x in f.split(",")])
        else:
            folds.append(int(f))
    sitk_files, mask_path, study_name, good_file_paths = main(
        model_path=args.model_path.strip(),
        series_paths=args.series_paths,
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

    if args.is_dicom is True:
        import pydicom_seg

        metadata_template = pydicom_seg.template.from_dcmqi_metainfo(
            args.metadata_path
        )
        writer = pydicom_seg.MultiClassWriter(
            template=metadata_template,
            skip_empty_slices=True,
            skip_missing_segment=False,
        )

        dcm = writer.write(mask, good_file_paths[0])
        # output_dcm_path = f"{args.output_dir}/{study_name}.dcm"
        output_dcm_path = f"{args.output_dir}/prediction.dcm"
        print(f"writing dicom output to {output_dcm_path}")
        dcm.save_as(output_dcm_path)

        if args.rt_struct_output:
            rt_struct_output = f"{args.output_dir}/struct.dcm"
            print(f"writing dicom struct to {rt_struct_output}")
            from rtstruct_writers import save_mask_as_rtstruct

            mask_array = np.transpose(sitk.GetArrayFromImage(mask), [1, 2, 0])
            with open(args.metadata_path) as o:
                metadata = json.load(o)
            segment_info = [
                [
                    element["SegmentDescription"],
                    element["recommendedDisplayRGBValue"],
                ]
                for element in metadata["segmentAttributes"][0]
            ]
            save_mask_as_rtstruct(
                mask_array,
                os.path.dirname(good_file_paths[0][0]),
                output_path=rt_struct_output,
                segment_info=segment_info,
            )

    if args.proba_map is True:
        class_idx = 1

        input_proba_map = f"{args.output_dir}/volume.npz"
        output_proba_map = f"{args.output_dir}/probabilities.nii.gz"
        input_file = sitk.ReadImage(sitk_files[0])
        proba_map = sitk.GetImageFromArray(
            np.load(input_proba_map)["probabilities"][class_idx]
        )
        proba_map.CopyInformation(input_file)
        threshold = sitk.ThresholdImageFilter()
        threshold.SetLower(0.1)
        threshold.SetUpper(1.0)
        proba_map = threshold.Execute(proba_map)
        print(f"writing probability map to {output_proba_map}")
        sitk.WriteImage(proba_map, output_proba_map)

        if args.is_dicom is True:
            from pydicom_seg_writers import FractionalWriter

            metadata_template = pydicom_seg.template.from_dcmqi_metainfo(
                args.metadata_path
            )
            writer = FractionalWriter(
                template=metadata_template,
                skip_empty_slices=True,
                skip_missing_segment=False,
            )

            dcm = writer.write(proba_map, good_file_paths[0])
            output_dcm_path = f"{args.output_dir}/probabilities.dcm"
            print(f"writing dicom output to {output_dcm_path}")
            dcm.save_as(output_dcm_path)

    if args.save_nifti_inputs is True:
        for sitk_file in sitk_files:
            basename = os.path.basename(sitk_file)
            for s in [".mha", ".nii.gz"]:
                basename = basename.rstrip(s)
            output_nifti = f"{args.output_dir}/{basename}.nii.gz"
            print(f"Copying Nifti to {output_nifti}")
            sitk.WriteImage(sitk.ReadImage(sitk_file), output_nifti)
