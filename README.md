# Docker container-ready nnUNet wrapper for SITK-readable and DICOM files

## Context

Given that [nnUNet](https://github.com/MIC-DKFZ/nnUNet) is a relatively flexible framework, we have developed a container that allows users to run nnUNet in a container while varying the necessary models. The main features are inferring all necessary parameters from the nnUNet files (spacing, extensions) and working for both DICOM folder and SITK-readable files. If the input is a DICOM, the segmentation is converted into a DICOM-seg file, compatible with PACS systems.

## Usage 

### Standalone script

A considerable objective of this framework was its deployment as a standalone tool (for `bash`). To use it:

1. Install the necessary packages using an appropriate Python environment (i.e. `pip install -r requirements.txt`). We have tested this using Python `v3.11`
2. Give executable permissions to ./entrypoint.sh (i.e. `chwon +x entrypoint.sh`)
3. Run the `entrypoint.sh` script with the appropriate arguments and flags. To know what they are please run `./entrypoint.sh -h`, i.e.

```bash
$ ./entrypoint.sh -h
```

```
script to run an nnUNet model while handling input and output conversions automatically.
spacing and extension are inferred directly from the nnUNet model.

usage:
    ./entrypoint -i INPUT_PATHS -o OUTPUT_FOLDER -m MODEL_FOLDER [-d] [-M METADATA]

args:
    -i      input paths to different files or DICOM folders (CANNOT CONTAIN SPACE CHARACTERS)
    -o      path to output folder
    -m      folder for the nnUNet model (should contain folds_X, plans.json, dataset.json, etc.)
    -d      (optional) tells the ./entrypoint.sh that the input is a DICOM folder
    -M      (optional) metadata template file for volume to DICOM-seg conversion. Must be specified
            if input is DICOM (if -d is used). This template can be generated using the following tool:
            https://qiicr.org/dcmqi/#/seg

```

#### Example

```
./entrypoint.sh \
    -i dicom_dataset/dicom_study/dicom_series_1 dicom_dataset/dicom_study/dicom_series_2 \
    -o output/ \
    -m nnunet_model_folder \
    -d \
    -M metadata_templates/anatomical-region-of-interest.json
```

As shown above, it is possible to provide more than one series to this script, which assumes that the order by which each series is provided corresponds to the series annotation typical of nnUNet (i.e. `0000` and `0001` for the first and second series types, respectively).

### Running as a Docker container

Firstly, users must install [Docker](https://www.docker.com/). **Docker requires `sudo` access so users should be sure to have this**. Then:

1. Build the container (`sudo docker build -f Dockerfile . -t nnunet_predict`)
2. Run the container. We have replicated this as an additional script (`./entrypoint-with-docker.sh`) with the same arguments as those specified to run as a standalone tool with the addition of a `-I` flag specifying the name of the Docker image.

With `entrypoint-with-docker.sh`, this:

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
sudo ./entrypoint-with-docker.sh \
    -i $INPUT_PATHS \
    -o $OUTPUT_FOLDER \
    -m $MODEL_FOLDER \
    -d \
    -M $METADATA_TEMPLATE \
    -I $DOCKER_IMAGE
```

#### Notes regarding use as container

Docker containers typically have a small overhead but are better if you are managing multiple package installations and versions. So use the one that makes the most sense for you.