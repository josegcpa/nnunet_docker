import argparse
import os
import SimpleITK as sitk
from pathlib import Path

supported_extensions = [
    "mha","nii.gz","nii"
]

def process_file_name(file_path:str):
    file_parts = file_path.split(os.sep)
    file_root = file_parts[-1]
    folder_path = os.sep.join(file_parts[:-1])
    for extension in supported_extensions:
        if file_root[-len(extension):] == extension:
            return folder_path,file_root[:-(len(extension)+1)],extension

def resample_to_target(moving_image:sitk.Image,
                       target_image:sitk.Image):
    return sitk.Resample(moving_image,target_image,
                         sitk.Transform(),sitk.sitkBSpline,0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts volumes to desired format and adds the necessary sequence codes")

    parser.add_argument(
        "--input_paths",dest="input_paths",required=True,nargs="+",
        help="Paths to input file")
    parser.add_argument(
        "--output_folder",dest="output_folder",required=True,
        help="Folder where output is stored (created if unavailable)")
    parser.add_argument(
        "--output_extension",dest="output_extension",default="nii.gz",
        help="Output extension",choices=["mha","nii","nii.gz"])
    parser.add_argument(
        "--sequence_codes",dest="sequence_codes",nargs="+",
        help="Sequence code. Must be four digit number",
        default=None)
    
    args = parser.parse_args()

    if args.sequence_codes is None:
        sequence_codes = [str(seq).rjust(4,"0") 
                          for seq in range(len(args.input_paths))]
    if len(sequence_codes) != len(args.input_paths):
        raise "sequence_codes and input_paths should have the same length"

    folder_path,file_root,extension = process_file_name(
        args.input_paths[0])
        
    images = [sitk.ReadImage(x) for x in args.input_paths]
    # resample to space of first image
    if len(images) > 1:
        images[1:] = [resample_to_target(x,images[0])
                      for x in images[1:]]
    
    for image,sequence_code in zip(images,sequence_codes):
        output_path = os.path.join(
            args.output_folder,
            f"{file_root}_{sequence_code}.{args.output_extension}")
        Path(output_path).parent.mkdir(parents=True,exist_ok=True)
        sitk.WriteImage(image,output_path)