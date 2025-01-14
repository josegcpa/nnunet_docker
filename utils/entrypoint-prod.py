import os
import SimpleITK as sitk
import time
import requests
import json

from utils import (
    export_proba_map_and_mask,
    export_dicom_files,
    make_parser,
)
from entrypoint import main


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.count = 0

    def print_time(self, extra_str=None):
        elapsed_time = time.time() - self.start_time
        if extra_str is None:
            print(f"Elapsed time {self.count}: {elapsed_time} seconds")
        else:
            print(
                f"Elapsed time {self.count} ({extra_str}): \
                  {elapsed_time} seconds"
            )
        self.count += 1


def main_args(args):
    timer = Timer()
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

    timer.print_time()

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

    timer.print_time()

    if args.save_nifti_inputs is True:
        for sitk_file in sitk_files:
            basename = os.path.basename(sitk_file)
            for s in [".mha", ".nii.gz"]:
                basename = basename.rstrip(s)
            output_nifti = f"{args.output_dir}/{basename}.nii.gz"
            print(f"Copying Nifti to {output_nifti}")
            sitk.WriteImage(sitk.ReadImage(sitk_file), output_nifti)

        timer.print_time()

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
        return "empty"

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

        timer.print_time()

    return args.success_message


if __name__ == "__main__":
    parser = make_parser()

    parser.add_argument(
        "--job_id",
        default=None,
        help="Job ID that will be used to post job status/create log file",
        type=str,
    )
    parser.add_argument(
        "--update_url",
        default=None,
        help="URL to be used to post job status",
        type=str,
    )
    parser.add_argument(
        "--success_message",
        default="done",
        help="Message to be posted in case of success",
        type=str,
    )
    parser.add_argument(
        "--failure_message",
        default="failed",
        help="Message to be posted in case of failure",
        type=str,
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to log file (with job_id, and success/failure messages)",
        type=str,
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enters debug mode",
    )

    args = parser.parse_args()

    if args.debug is not True:
        try:
            log_out = main_args(args)
            status = args.success_message
            err = ""
        except KeyboardInterrupt:
            status = args.failure_message
            log_out = ""
            err = "User interrupted execution"
        except Exception as e:
            status = args.failure_message
            log_out = ""
            err = repr(e)
            print(e)
    else:
        log_out = main_args(args)
        status = args.success_message
        err = ""

    if "empty" in str(log_out):
        status = status + "_no_output"
        log_txt = "No lesion detected"
    else:
        log_txt = ""
    data = {
        "job_id": args.job_id,
        "status": status,
        "output_log": log_txt,
        "output_error": err,
    }

    if (args.update_url is not None) and (args.update_url != "skip"):
        print(f"Posting {status} to {args.update_url} for job {args.job_id}")
        requests.post(
            f"{args.update_url}",
            data=data,
        )

    if args.log_file is not None:
        if os.path.exists(args.log_file):
            with open(args.log_file, "r") as f:
                log_out = json.load(f)
            for k in ["status", "output_log", "output_error"]:
                log_out[k] = data[k]
        else:
            log_out = data
        with open(args.log_file, "w") as f:
            json.dump(log_out, f)

    if len(err) > 0:
        raise Exception(err)
