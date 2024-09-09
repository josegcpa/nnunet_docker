# based on the TotalSegmentator implementation

import numpy as np
import logging
from tqdm import tqdm
from rt_utils import RTStructBuilder


def save_mask_as_rtstruct(
    img_data: np.ndarray,
    dcm_reference_file: str,
    output_path: str,
    segment_info: tuple[str, list[int]],
):
    logging.basicConfig(level=logging.WARNING)  # avoid messages from rt_utils

    # create new RT Struct - requires original DICOM
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_reference_file)

    # retrieve selected classes
    selected_classes = np.unique(img_data)
    selected_classes = selected_classes[selected_classes > 0].tolist()
    if len(selected_classes) == 0:
        return None

    # add mask to RT Struct
    for class_idx in tqdm(selected_classes):
        class_name, class_colour = segment_info[class_idx - 1]
        binary_img = img_data == class_idx
        if binary_img.sum() > 0:  # only save none-empty images

            # rotate nii to match DICOM orientation
            binary_img = np.rot90(binary_img, 1, (0, 1))  # rotate segmentation in-plane

            # add segmentation to RT Struct
            rtstruct.add_roi(
                mask=binary_img,  # has to be a binary numpy array
                name=class_name,
                color=class_colour,
            )

    rtstruct.save(str(output_path))
