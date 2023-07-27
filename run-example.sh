# explanation:
#   * `--gpus all`` gives docker access to GPUs
#   * `--user "$(id -u):$(id -g)"` avoids ownership issues with docker created files
#   * `-v $(realpath example):/data/input` mounts the input folder in /data/input
#   * `-v $(realpath tmp_output):/data/output` mounts the output folder in /data/output
#   * `-v $(realpath nnUNetTrainer__nnUNetPlans__3d_fullres):/model` mounts the model volume in /model
#   * `-v $(realpath metadata_templates):/metadata nnunet_predict` mounts the metadata template directory
#   * `nnunet_predict` name of container
#   * `-i /data/input/image_T2.mha` further specifies image name inside of /data/input and metadata template

docker run \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -v $(realpath example/example_real/t2axial_dicoms/2015050934_RM_PELVICA):/data/input \
    -v $(realpath tmp_output):/data/output \
    -v $(realpath models/nnUNetTrainer__nnUNetPlans__3d_fullres):/model \
    -v $(realpath metadata_templates):/metadata nnunet_predict \
    nnunet_predict \
    -i /data/input/MR_T2W_TSE_ax -d -M /metadata/whole-prostate.json
