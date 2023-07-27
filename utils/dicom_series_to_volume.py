"""
Converts DICOM series to volumes (nii, nii.gz, mha).
"""

__version__ = '0.1'
__author__ = 'José Guilherme de Almeida'

import argparse
import SimpleITK as sitk
from pathlib import Path
from .utils import resample_image
from .utils import read_dicom_as_sitk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts DICOM series to volume (nii, nii.gz, mha)")
    
    parser.add_argument(
        "--input_path",required=True,
        help="Path to folder containing files.")
    parser.add_argument(
        "--output_path",required=True,
        help="Path to output.")
    parser.add_argument(
        "--spacing",type=float,default=None,nargs="+",
        help="Sets output spacing")
    
    args = parser.parse_args()
    
    sitk_image = read_dicom_as_sitk(
        Path(args.input_path).glob("*dcm"))
    
    if isinstance(sitk_image,str):
        raise Exception(
            "Failed to open series. Reason: {}".format(sitk_image))
        
    Path(args.output_path).parent.mkdir(exist_ok=True,parents=True)
    if args.spacing is not None:
        sitk_image = resample_image(sitk_image,out_spacing=args.spacing)
    sitk.WriteImage(sitk_image,args.output_path)
