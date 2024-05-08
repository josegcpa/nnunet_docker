if __name__ == "__main__":
    import argparse
    import os
    import docker

    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--container_name",
        "-c",
        dest="container_name",
        help="Name of container to use",
        default="nnunet_predict",
        required=False,
    )

    parser.add_argument(
        "--series_paths",
        "-i",
        dest="series_paths",
        nargs="+",
        help="Path to input series",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        "-m",
        dest="model_path",
        help="Path to nnUNet model folder",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_name",
        "-ckpt",
        dest="checkpoint_name",
        help="Checkpoint name for nnUNet",
        default="checkpoint_best.pth",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        help="Path to output directory",
        required=True,
    )
    parser.add_argument(
        "--metadata_path",
        "-M",
        dest="metadata_path",
        help="Path to metadata template for DICOM-Seg output",
        required=True,
    )
    parser.add_argument(
        "--study_uid",
        "-s",
        dest="study_uid",
        help="Study UID if series are SimpleITK-readable files",
        default=None,
    )
    parser.add_argument(
        "--folds",
        "-f",
        dest="folds",
        help="Sets which folds should be used with nnUNet",
        nargs="+",
        type=str,
        default=("0",),
    )
    parser.add_argument(
        "--tta",
        "-t",
        dest="tta",
        help="Uses test-time augmentation during prediction",
        action="store_true",
    )
    parser.add_argument(
        "--proba_map",
        "-p",
        help="Produces a Nifti format probability map",
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
        help="Moves Nifti inputs to output folder",
        action="store_true",
    )
    parser.add_argument(
        "--tmp_dir", dest="tmp_dir", help="Temporary directory", default=".tmp"
    )
    parser.add_argument(
        "--is_dicom",
        "-D",
        dest="is_dicom",
        help="Assumes input is DICOM (and also converts to DICOM seg)",
        action="store_true",
    )

    args = parser.parse_args()

    client = docker.from_env()

    data_inputs_in_docker = []
    volumes = {}
    folds = args.folds
    for series_path in args.series_paths:
        series_folder_name = os.path.basename(series_path.rstrip(os.sep))
        data_input_in_docker = f"/data/input/{series_folder_name}"
        data_inputs_in_docker.append(data_input_in_docker)
        volumes[
            os.path.abspath(
                series_path,
            )
        ] = {"bind": data_input_in_docker, "mode": "ro"}

    if "," in folds[0]:
        # some applications do not allow for space separated arguments
        # so comma separation also possible
        folds = list(set(folds[0].split(",")))

    volumes[os.path.abspath(args.output_dir)] = {
        "bind": "/data/output",
        "mode": "rw",
    }

    if os.path.isdir(os.path.abspath(args.model_path)):
        volumes[os.path.abspath(args.model_path)] = {
            "bind": "/model",
            "mode": "ro",
        }
    metadata_dir = os.path.abspath(os.path.dirname(args.metadata_path))
    metadata_name = os.path.basename(args.metadata_path)
    volumes[metadata_dir] = {
        "bind": "/metadata",
        "mode": "ro",
    }

    user = f"{os.getuid()}:{os.getgid()}"

    complete_command = [
        f"--series_paths {' '.join(data_inputs_in_docker)}",
        f"--metadata_path /metadata/{metadata_name}",
        f"--folds {' '.join(folds)}",
        f"--checkpoint_name {args.checkpoint_name}",
    ]
    if args.study_uid is not None:
        complete_command.append(f"--study_uid {args.study_uid}")
    if args.is_dicom is True:
        complete_command.append("--is_dicom")
    if args.tta is True:
        complete_command.append("--tta")
    if args.proba_map is True:
        complete_command.append("--proba_map")
    if args.save_nifti_inputs is True:
        complete_command.append("--save_nifti_inputs")
    if args.rt_struct_output is True:
        complete_command.append("--rt_struct_output")

    docker_command = [
        "docker run",
        " ".join(
            [
                f"-v {k}:{volumes[k]['bind']}:{volumes[k]['mode']}"
                for k in volumes
            ]
        ),
        f"--user {user}",
        "--rm",
        "--entrypoint ''",
        "--gpus all",
        args.container_name,
        " ".join(complete_command),
    ]

    print(" ".join(docker_command))

    output = client.containers.run(
        args.container_name,
        " ".join(complete_command),
        volumes=volumes,
        auto_remove=True,
        stderr=True,
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=["1"], capabilities=[["gpu"]]
            )
        ],
    )

    print(output.decode())
