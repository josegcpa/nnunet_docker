# Docker container-ready nnUNet wrapper for SITK-readable and DICOM files

## Context

Given that [nnUNet](https://github.com/MIC-DKFZ/nnUNet) is a relatively flexible framework, we have developed a container that allows users to run nnUNet in a container while varying the necessary models. The main features are inferring all necessary parameters from the nnUNet files (spacing, extensions) and working for both DICOM folder and SITK-readable files. If the input is a DICOM, the segmentation is converted into a DICOM-seg file, compatible with PACS systems.

## Usage 

### Standalone script

A considerable objective of this framework was its deployment as a standalone tool (for `bash`). To use it:

1. Install the necessary packages using an appropriate Python environment (i.e. `pip install -r requirements.txt`). We have tested this using Python `v3.11`
2. Run `python utils/entrypoints.py --help` to see the available options
3. Segment away!

```bash
python utils/entrypoint.py --help
```

```
usage: Entrypoint for nnUNet prediction. Handles all data format conversions. [-h] --series_paths SERIES_PATHS [SERIES_PATHS ...] --model_path
                                                                              MODEL_PATH [--checkpoint_name CHECKPOINT_NAME] --output_dir
                                                                              OUTPUT_DIR --metadata_path METADATA_PATH [--study_uid STUDY_UID]
                                                                              [--folds FOLDS [FOLDS ...]] [--tta] [--tmp_dir TMP_DIR] [--is_dicom]
                                                                              [--proba_map] [--rt_struct_output] [--save_nifti_inputs]

options:
  -h, --help            show this help message and exit
  --series_paths SERIES_PATHS [SERIES_PATHS ...], -i SERIES_PATHS [SERIES_PATHS ...]
                        Path to input series
  --model_path MODEL_PATH, -m MODEL_PATH
                        Path to nnUNet model folder
  --checkpoint_name CHECKPOINT_NAME, -ckpt CHECKPOINT_NAME
                        Checkpoint name for nnUNet
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to output directory
  --metadata_path METADATA_PATH, -M METADATA_PATH
                        Path to metadata template for DICOM-Seg output
  --study_uid STUDY_UID, -s STUDY_UID
                        Study UID if series are SimpleITK-readable files
  --folds FOLDS [FOLDS ...], -f FOLDS [FOLDS ...]
                        Sets which folds should be used with nnUNet
  --tta, -t             Uses test-time augmentation during prediction
  --tmp_dir TMP_DIR     Temporary directory
  --is_dicom, -D        Assumes input is DICOM (and also converts to DICOM seg; prediction.dcm in output_dir)
  --proba_map, -p       Produces a Nifti format probability map (probabilities.nii.gz in output_dir)
  --rt_struct_output    Produces a DICOM RT Struct file (struct.dcm in output_dir)
  --save_nifti_inputs, -S
                        Moves Nifti inputs to output folder (volume_XXXX.nii.gz in output_dir)```

Example:

```bash
python utils/entrypoints.py \
    -i study/series_1 study/series_2 study/series_3 \
    -o example_output/ \
    -m models/prostate_model \
    -M metadata_templates/metadata-template.json \
    -D -f 0 1 2 3 4 \
    --proba_map \
    --save_nifti_inputs
```

### Running as a Docker container

Firstly, users must install [Docker](https://www.docker.com/). **Docker requires `sudo` access so users should be sure to have this**. Then:

1. Build the container (`sudo docker build -f Dockerfile . -t nnunet_predict`)
2. Run the container. We have replicated this as an additional script (`utils/entrypoint-with-docker.py`) with the same arguments as those specified to run as a standalone tool with the addition of a `-c` flag specifying the name of the Docker image.

With `utils/entrypoint-with-docker.py`, this:

```
docker run \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -v $(dirname $(realpath $INPUT_PATHS)):/data/input \
    -v $(realpath $OUTPUT_FOLDER):/data/output \
    -v $(realpath $MODEL_FOLDER):/model \
    -v $(dirname $(realpath $METADATA_TEMPLATE)):/metadata \
    --rm \
    $DOCKER_IMAGE \
    -i $file_names_in_docker -d -M $metadata_name_in_docker
```

becomes this (for a DICOM input):

```
python utils/entrypoint-with-docker.py \
    -i $INPUT_PATHS \
    -o $OUTPUT_FOLDER \
    -m $MODEL_FOLDER \
    -d \
    -M $METADATA_TEMPLATE \
    -c $DOCKER_IMAGE
```

### Notes on using DICOM

It is necessary to generate metadata templates for the conversion between the segmentation prediction volume and DICOM volumes. To generate these, the `pydicom_seg` developers recommend [this web app](https://qiicr.org/dcmqi/#/seg). It is easy to use and generates reliable metadata templates. Metadata templates should be generated for all segmentation targets to ensure that everything is correctly formatted.