#!/bin/bash

set -e

TMP_FOLDER=.tmp

METADATA_TEMPLATE=template.json
OUTPUT_FOLDER=/data/output
MODEL_FOLDER=model
IS_DICOM=0
while getopts "i:o:e:m:I:M:s:n:dh" opt; do
  case ${opt} in
    i )
       INPUT_PATHS=($OPTARG)
       ;;
    o )
       OUTPUT_FOLDER=$OPTARG
       ;;
    m )
       MODEL_FOLDER=$OPTARG
       ;;
    I )
       DOCKER_IMAGE=$OPTARG
       ;;
    M )
       METADATA_TEMPLATE=$OPTARG
       ;;
    h )
       cat assets/helptext-docker.txt
       exit 0
       ;;
    d ) 
       IS_DICOM=1
       ;;
  esac
done

echo "Running nnUNet wrapper with:"
echo "    input paths: ${INPUT_PATHS[@]}"
echo "    output folder: $OUTPUT_FOLDER"
echo "    model folder: $MODEL_FOLDER"
echo "    docker image: $DOCKER_IMAGE"
echo "    is dicom: $IS_DICOM"
echo "    spacing (from nnunet model folder): $SPACING"
echo "    extension (from nnunet model folder): $EXTENSION"

file_names_in_docker=$(
   for file in ${INPUT_PATHS[@]}; 
   do 
      echo /data/input/$(basename $file); 
   done | xargs)
metadata_name_in_docker=/metadata/$(basename $METADATA_TEMPLATE)

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
