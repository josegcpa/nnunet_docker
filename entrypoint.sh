#!/bin/bash

set -e

TMP_FOLDER=.tmp
SEP=assets/separator.txt

METADATA_TEMPLATE=template.json
OUTPUT_FOLDER=/data/output
MODEL_FOLDER=model
DISABLE_TTA=""
FOLDS="0,1,2,3,4"
IS_DICOM=0
while getopts "i:o:m:M:f:dhD" opt; do
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
    d ) 
       IS_DICOM=1
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

echo "" 
cat $SEP
if [[ $IS_DICOM == 1 ]]
then
   echo "Converting DICOM to nifti..."
   i=0
   for file in ${INPUT_PATHS[@]}
   do
      output_path=$TMP_FOLDER/raw/placeholder_$(printf %04d $i).$EXTENSION
      python -m utils.dicom_series_to_volume \
         --input_path "$file" \
         --output_path "$output_path"
      i=$(($i+1))
   done
   ls $TMP_FOLDER
else
   echo "Fixing input file names..."
   python -m utils.prepare_for_nnunet \
      --input_paths ${INPUT_PATHS[@]} \
      --output_folder $TMP_FOLDER/raw \
      --output_extension $EXTENSION
fi
cat $SEP

echo ""

cat $SEP
echo "Running nnUNet..."
export nnUNet_raw=$TMP_FOLDER/raw
export nnUNet_preprocessed=$TMP_FOLDER/preprocessed
export nnUNet_results=$MODEL_FOLDER
nnUNetv2_predict_from_modelfolder \
    -i $TMP_FOLDER/raw \
    -o $OUTPUT_FOLDER \
    -m $MODEL_FOLDER $DISABLE_TTA -f $(echo $FOLDS | tr "," " ")
cat $SEP

echo ""

cat $SEP
post_processing_pkl=$MODEL_FOLDER/crossval_results_folds_0_1_2_3_4/postprocessing.pkl
if [[ -f "$post_processing_pkl" ]]
then
   echo "Running postprocessing..."
   nnUNetv2_apply_postprocessing \
      -i $OUTPUT_FOLDER \
      -o $OUTPUT_FOLDER \
      -pp_pkl_file $post_processing_pkl \
      -np 8 \
      -plans_json $MODEL_FOLDER/crossval_results_folds_0_1_2_3_4/plans.json
else
   echo "Postprocessing not applied ($post_processing_pkl not found)"
fi
cat $SEP

echo ""

cat $SEP
if [[ $IS_DICOM == 1 ]]
then
   echo "Converting volume to DICOM-seg..."
   python utils/volume_to_dicom_seg.py \
      --mask_path $OUTPUT_FOLDER/placeholder.$EXTENSION \
      --source_data_path ${INPUT_PATHS[0]} \
      --metadata_path $METADATA_TEMPLATE \
      --output_path $OUTPUT_FOLDER/placeholder.dcm
fi
cat $SEP