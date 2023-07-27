import argparse
import json
import os
import SimpleITK as sitk
import pydicom_seg
from pydicom import dcmread
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask_path",dest="mask_path",
        help="Path to mask in SITK-readable format")
    parser.add_argument(
        "--source_data_path",dest="source_data_path",
        help="Path to DICOM folder containing source DICOM image")
    parser.add_argument(
        "--metadata_path",dest="metadata_path",
        help="Path to metadata file. DICOM-seg recommends \
            https://qiicr.org/dcmqi/#/seg to generate these files")
    parser.add_argument(
        "--output_path",dest="output_path",
        help="Path to output")

    args = parser.parse_args()

    mask = sitk.ReadImage(args.mask_path)
    
    metadata_template = pydicom_seg.template.from_dcmqi_metainfo(
        args.metadata_path)
    writer = pydicom_seg.MultiClassWriter(
        template=metadata_template,
        skip_empty_slices=True,
        skip_missing_segment=False,)

    dcm = writer.write(mask,
                       glob(os.path.join(args.source_data_path,"*")))
    dcm.save_as(args.output_path)
