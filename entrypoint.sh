#!/bin/bash

time_start=$(date +%s)
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
END_COL='\033[0m'
separate() {
   SEP=$(cat assets/separator-end.txt)
   mode=$1
   input_str="$2"
   start_str="Starting:"
   completed_str="Completed:"
   length=$((${#start_str} + ${#input_str} + 1))
   SEP=${SEP:1:$length}
   echo $SEP
   if [[ "$mode" == "start" ]]
   then
      printf "$RED$start_str $input_str$END_COL\n"
   else
      printf "$GREEN$completed_str $input_str$END_COL\n"
   fi
   echo $SEP
   echo ""
}

TMP_FOLDER=.tmp

METADATA_TEMPLATE=template.json
OUTPUT_FOLDER=/data/output
MODEL_FOLDER=model
DISABLE_TTA=""
FOLDS="0,1,2,3,4"
OUTPUT_NAME=placeholder
DEVICE="cuda"
IS_DICOM=0
while getopts "i:o:m:M:V:f:n:dhD" opt; do
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
    M )
       METADATA_TEMPLATE=$OPTARG
       ;;
    f )
       FOLDS=$OPTARG
       ;;
    D )
       DISABLE_TTA="--disable_tta"
       ;;
    V )
       DEVICE=$OPTARG
       ;;
    d ) 
       IS_DICOM=1
       ;;
    n ) 
       OUTPUT_NAME=$OPTARG
       ;;
    h )
       cat assets/helptext.txt
       exit 0
       ;;
  esac
done

EXTENSION=$(python utils/retrieve-extension.py --model_folder $MODEL_FOLDER)
SPACING=$(python utils/retrieve-spacing.py --model_folder $MODEL_FOLDER)

echo "Running nnUNet wrapper with:"
echo "    input paths: ${INPUT_PATHS[@]}"
echo "    output folder: $OUTPUT_FOLDER"
echo "    model folder: $MODEL_FOLDER"
echo "    is dicom: $IS_DICOM"
echo "    spacing (from nnunet model folder): $SPACING"
echo "    extension (from nnunet model folder): $EXTENSION"

mkdir -p $TMP_FOLDER
mkdir -p $TMP_FOLDER/raw
mkdir -p $TMP_FOLDER/preprocessed

if [[ $IS_DICOM == 1 ]]
then
   separate start "converting DICOM to nifti"
   i=0
   for file in ${INPUT_PATHS[@]}
   do
      output_path=$TMP_FOLDER/raw/"$OUTPUT_NAME"_$(printf %04d $i).$EXTENSION
      python -m utils.dicom_series_to_volume \
         --input_path "$file" \
         --output_path "$output_path"
      i=$(($i+1))
   done
   separate complete "converting DICOM to nifti"
else
   separate start "fixing input names"
   python -m utils.prepare_for_nnunet \
      --input_paths ${INPUT_PATHS[@]} \
      --output_folder $TMP_FOLDER/raw \
      --output_extension $EXTENSION
   separate complete "fixing input names"
fi

separate start "running nnUNet"
export nnUNet_raw=$TMP_FOLDER/raw
export nnUNet_preprocessed=$TMP_FOLDER/preprocessed
export nnUNet_results=$MODEL_FOLDER
nnUNetv2_predict_from_modelfolder \
    -device $DEVICE \
    -i $TMP_FOLDER/raw \
    -o $OUTPUT_FOLDER \
    -m $MODEL_FOLDER $DISABLE_TTA -f $(echo $FOLDS | tr "," " ")
separate complete "running nnUNet"

post_processing_pkl="$MODEL_FOLDER/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"
if [[ -f "$post_processing_pkl" ]]
then
   separate start "running postprocessing"
   nnUNetv2_apply_postprocessing \
      -i $OUTPUT_FOLDER \
      -o $OUTPUT_FOLDER \
      -pp_pkl_file $post_processing_pkl \
      -np 8 \
      -plans_json $MODEL_FOLDER/crossval_results_folds_0_1_2_3_4/plans.json
   separate complete "running postprocessing"
else
   ls $MODEL_FOLDER
   ls $MODEL_FOLDER/crossval_results_folds_0_1_2_3_4
   echo "Postprocessing not applied \($post_processing_pkl not found\)"
fi

if [[ $IS_DICOM == 1 ]]
then
   separate start "converting volume to DICOM-seg"
   python utils/volume_to_dicom_seg.py \
      --mask_path $OUTPUT_FOLDER/$OUTPUT_NAME.$EXTENSION \
      --source_data_path ${INPUT_PATHS[0]} \
      --metadata_path $METADATA_TEMPLATE \
      --output_path $OUTPUT_FOLDER/$OUTPUT_NAME.dcm
   separate complete "converting volume to DICOM-seg"
fi

time_end=$(date +%s)
echo "Time elapsed:" $(($time_end - $time_start)) seconds