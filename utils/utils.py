import os
import numpy as np
import SimpleITK as sitk
import pydicom_seg
import json
from glob import glob
from typing import List, Dict
from pydicom import dcmread
from scipy import ndimage

from typing import Sequence


def filter_by_bvalue(
    dicom_files: list, target_bvalue: int, exact: bool = False
) -> list:
    """
    Selects the DICOM values with a b-value which is exactly or closest to
    target_bvalue (depending on whether exact is True or False).

    Args:
        dicom_files (list): list of pydicom file objects.
        target_bvalues (int): the expected b-value.
        exact (bool, optional): whether the b-value matching is to be exact
            (raises error if exact target_bvalue is not available) or
            approximate returns the b-value which is closest to target_bvalue.

    Returns:
        list: list of b-value-filtered pydicom file objects.
    """
    BVALUE_TAG = ("0018", "9087")
    SIEMENS_BVALUE_TAG = ("0019", "100c")
    GE_BVALUE_TAG = ("0043", "1039")
    bvalues = []
    for d in dicom_files:
        curr_bvalue = None
        bvalue = d.get(BVALUE_TAG, None)
        siemens_bvalue = d.get(SIEMENS_BVALUE_TAG, None)
        ge_bvalue = d.get(GE_BVALUE_TAG, None)
        if bvalue is not None:
            curr_bvalue = bvalue.value
        elif siemens_bvalue is not None:
            curr_bvalue = siemens_bvalue.value
        elif ge_bvalue is not None:
            curr_bvalue = ge_bvalue.value
            if isinstance(curr_bvalue, bytes):
                curr_bvalue = curr_bvalue.decode()
            curr_bvalue = str(curr_bvalue)
            if "[" in curr_bvalue and "]" in curr_bvalue:
                curr_bvalue = (
                    curr_bvalue.strip().strip("[").strip("]").split(",")
                )
                curr_bvalue = [int(x) for x in curr_bvalue]
            if isinstance(curr_bvalue, list) is False:
                curr_bvalue = curr_bvalue.split("\\")
                curr_bvalue = str(curr_bvalue[0])
            else:
                curr_bvalue = str(curr_bvalue[0])
            if len(curr_bvalue) > 5:
                curr_bvalue = curr_bvalue[-4:]
        if curr_bvalue is None:
            curr_bvalue = 0
        bvalues.append(int(curr_bvalue))
    unique_bvalues = set(bvalues)
    if len(unique_bvalues) in [0, 1]:
        return dicom_files
    if (target_bvalue not in unique_bvalues) and (exact is True):
        raise RuntimeError("Requested b-value not available")
    best_bvalue = sorted(unique_bvalues, key=lambda b: abs(b - target_bvalue))[
        0
    ]
    dicom_files = [f for f, b in zip(dicom_files, bvalues) if b == best_bvalue]
    return dicom_files


def resample_image_to_target(
    moving: sitk.Image,
    target: sitk.Image,
    is_label: bool = False,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to the space of a target image.

    Args:
      moving: The SimpleITK image to resample.
      target: The target SimpleITK image to match.
      is_label (bool): whether the moving image is a label mask.

    Returns:
      The resampled SimpleITK image matching the target properties.
    """
    if is_label:
        interpolation = sitk.sitkNearestNeighbor
    else:
        interpolation = sitk.sitkBSpline

    output = sitk.Resample(moving, target, sitk.Transform(), interpolation, 0)
    return output


def resample_image(
    sitk_image: sitk.Image,
    out_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    is_mask: bool = False,
) -> sitk.Image:
    """Resamples an SITK image to out_spacing. If is_mask is True, uses
    nearest neighbour interpolation. Otherwise, it uses B-splines.

    Args:
        sitk_image (sitk.Image): SITK image.
        out_spacing (Sequence, optional): target spacing for the image.
            Defaults to [1.0, 1.0, 1.0].
        is_mask (bool, optional): sets the interpolation to nearest neighbour.
            Defaults to False.

    Returns:
        sitk.Image: resampled SITK image.
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    out_size = [
        int(
            np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))
        ),
        int(
            np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))
        ),
        int(
            np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))
        ),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0.0)

    if is_mask is True:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    output = resample.Execute(sitk_image)

    return output


def mode(a: np.ndarray) -> int | float:
    """
    Calculates the mode of an array.

    Args:
        a (np.ndarray): a numpy array.

    Returns:
        int | float: the mode of a.
    """
    u, c = np.unique(a, return_counts=True)
    return u[np.argmax(c)]


def get_origin(positions: np.ndarray, z_axis: int = 2) -> np.ndarray:
    """
    Returns the origin position from an array of positions (minimum for a given
    z-axis).

    Args:
        positions (np.ndarray): array containing all the positions in a given
            set of arrays.
        z_axis (int, optional): index corresponding to the z-axis. Defaults to
            2.

    Returns:
        np.ndarray: origin of the array.
    """
    origin = positions[positions[:, z_axis].argmin()]
    return origin


def dicom_orientation_to_sitk_direction(
    orientation: Sequence[float],
) -> np.ndarray:
    """Converts the DICOM orientation to SITK orientation. Based on the
    nibabel code that does the same. DICOM uses a more economic encoding
    as one only needs to specify two of the three cosine directions as they
    are all orthogonal. SITK does the more verbose job of specifying all three
    components of the orientation.

    Args:
        orientation (Sequence[float]): DICOM orientation.

    Returns:
        np.ndarray: SITK (flattened) orientation.
    """
    # based on nibabel documentation
    orientation = np.array(orientation).reshape(2, 3).T
    R = np.eye(3)
    R[:, :2] = np.fliplr(orientation)
    R[:, 2] = np.cross(orientation[:, 1], orientation[:, 0])
    R_sitk = np.stack([R[:, 1], R[:, 0], -R[:, 2]], 1)
    return R_sitk.flatten().tolist()


def get_contiguous_arr_idxs(
    positions: np.ndarray, ranking: np.ndarray
) -> np.array:
    """
    Uses the ranking to find breaks in positions and returns the elements in
    L which belong to the first contiguous array. Assumes that positions is an
    array of positions (a few of which may be overlapping), ranking is the order
    by which each slice was acquired and d is a dict whose keys will be filtered
    according to this.

    Args:
        positions (np.ndarray): positions with shape [N,3].
        ranking (np.ndarray): ranking used to sort slices.

    Returns:
        np.ndarray: an index vector with the instance numbers of the slices to be kept.
    """
    if all([len(x) == 3 for x in positions]) is False:
        return None
    assert len(positions) == len(ranking)
    order = np.argsort(ranking)
    positions = positions[:, 2][order]
    if len(positions) == 1:
        return None
    p_diff = np.abs(np.round(np.diff(positions) / np.diff(ranking[order]), 1))
    m = mode(p_diff)
    break_points = np.where(np.logical_and(p_diff > 4.5, p_diff > m))[0] + 1
    if len(break_points) == 0:
        return ranking
    segments = np.zeros_like(positions)
    segments[break_points] = 1
    segments = segments.cumsum().astype(int)
    S, C = np.unique(segments, return_counts=True)
    si = S[C >= 8]
    if len(si) == 0:
        return None
    si = si.min()
    output_segment_idxs = ranking[order][segments == si]
    return output_segment_idxs


def read_dicom_as_sitk(file_paths: List[str], metadata: Dict[str, str] = {}):
    """
    Reads a DICOM file as SITK, using z-spacing to order slices in the
    volume.

    Args:
        file_paths (List[str]): list of file paths belonging to the same
            series.
        metadata (Dict[str,str]): sets as SITK metadata. Defaults to {}.

    Returns:
        sitk.Image: SimpleITK image made up of the dcms in file_paths.
    """

    fs = []
    good_file_paths = []
    orientation = None
    series_path = os.path.dirname(file_paths[0])
    for dcm_file in file_paths:
        f = dcmread(dcm_file)
        if (0x0020, 0x0037) in f:
            orientation = f[0x0020, 0x0037].value
        if (0x0020, 0x0032) in f:
            fs.append(f)
            good_file_paths.append(dcm_file)
    fs = filter_by_bvalue(fs, 1400)
    if orientation is None:
        raise RuntimeError(f"No orientation available for {series_path}")
    position = np.array([x[0x0020, 0x0032].value for x in fs])
    rankings = np.array([x[0x0020, 0x0013].value for x in fs])

    segment_selection = get_contiguous_arr_idxs(position, rankings)
    if segment_selection is not None:
        fs = [x for x in fs if x[0x0020, 0x0013].value in segment_selection]
    position = np.array([x[0x0020, 0x0032].value for x in fs])
    rankings = np.array([x[0x0020, 0x0013].value for x in fs])
    if segment_selection is None:
        raise RuntimeError(f"Positions are incorrect for {series_path}")
    orientation = list(map(float, orientation))
    orientation_sitk = dicom_orientation_to_sitk_direction(orientation)
    z_axis = 2
    real_position = np.matmul(
        position, np.array(orientation_sitk).reshape([3, 3])
    )
    z_position = np.sort(real_position[:, z_axis])
    z_spacing = np.median(np.diff(z_position))
    if np.isclose(z_spacing, 0) == True:
        raise RuntimeError(
            f"Incorrect z-spacing information for {series_path}."
            + "This may be due to multiple slices having identical positions"
        )
    pixel_spacing = [*f[0x0028, 0x0030].value, z_spacing]
    fs = sorted(fs, key=lambda x: float(x[0x0020, 0x0032].value[z_axis]))

    origin_sitk = get_origin(position, z_axis=z_axis)
    pixel_spacing_sitk = pixel_spacing
    try:
        pixel_data = np.stack(
            [f.pixel_array for f in fs],
        )
    except Exception as e:
        print(e)
        raise RuntimeError(f"Pixel data could not be read for {series_path}")

    sitk_image = sitk.GetImageFromArray(pixel_data)
    sitk_image.SetDirection(orientation_sitk)
    sitk_image.SetOrigin(origin_sitk)
    sitk_image.SetSpacing(pixel_spacing_sitk)

    for k in metadata:
        sitk_image.SetMetaData(k, metadata[k])

    return sitk_image, good_file_paths


def get_study_uid(dicom_dir: List[str]) -> str:
    """
    Returns the study UID field from a random file in dicom_dir.

    Args:
        dicom_dir (str): directory with dicom (.dcm) files.

    Returns:
        str: string corresponding to study UID.
    """
    dcm_files = glob(f"{dicom_dir}/*dcm")
    if len(dcm_files) == 0:
        raise RuntimeError(f"No dcm files in {dicom_dir}")
    return dcmread(dcm_files[0])[(0x0020, 0x000D)].value


def export_to_dicom_seg(
    mask: sitk.Image,
    metadata_path: str,
    file_paths: list[str],
    output_dir: str,
    output_file_name: str = "prediction",
) -> str:
    """
    Exports a SITK image mask as a DICOM segmentation object.

    Args:
        mask (sitk.Image): an SITK file object corresponding to a mask.
        metadata_path (str): path to metadata template file.
        file_paths (list[str]): list of DICOM file paths corresponding to the
            original series.
        output_dir (str): path to output directory.
        output_file_name (str, optional): output file name. Defaults to
            "prediction".

    Returns:
        str: "success" if the process was successful, "empty mask" if the SITK
            mask contained no values.
    """
    import pydicom_seg

    metadata_template = pydicom_seg.template.from_dcmqi_metainfo(
        metadata_path.strip()
    )
    writer = pydicom_seg.MultiClassWriter(
        template=metadata_template,
        skip_empty_slices=True,
        skip_missing_segment=False,
    )

    if sitk.GetArrayFromImage(mask).sum() == 0:
        return "empty mask"
    dcm = writer.write(mask, file_paths[0])
    output_dcm_path = f"{output_dir}/{output_file_name}.dcm"
    print(f"writing dicom output to {output_dcm_path}")
    dcm.save_as(output_dcm_path)
    return "success"


def export_to_dicom_struct(
    mask: sitk.Image,
    metadata_path: str,
    file_paths: list[str],
    output_dir: str,
    output_file_name: str = "struct",
) -> str:
    """
    Exports a SITK image mask as a DICOM struct object.

    Args:
        mask (sitk.Image): an SITK file object corresponding to a mask.
        metadata_path (str): path to metadata template file.
        file_paths (list[str]): list of DICOM file paths corresponding to the
            original series.
        output_dir (str): path to output directory.
        output_file_name (str, optional): output file name. Defaults to
            "prediction".

    Returns:
        str: "success" if the process was successful, "empty mask" if the SITK
            mask contained no values.
    """
    from rtstruct_writers import save_mask_as_rtstruct

    rt_struct_output = f"{output_dir}/{output_file_name}.dcm"
    print(f"writing dicom struct to {rt_struct_output}")

    mask_array = np.transpose(sitk.GetArrayFromImage(mask), [1, 2, 0])
    if mask_array.sum() == 0:
        return "empty mask"

    with open(metadata_path.strip()) as o:
        metadata = json.load(o)
    segment_info = [
        [
            element["SegmentDescription"],
            element["recommendedDisplayRGBValue"],
        ]
        for element in metadata["segmentAttributes"][0]
    ]
    save_mask_as_rtstruct(
        mask_array,
        os.path.dirname(file_paths[0][0]),
        output_path=rt_struct_output,
        segment_info=segment_info,
    )

    return "success"


def export_proba_map(
    sitk_files: list[str],
    output_dir: str,
    proba_threshold: float | None = 0.1,
    min_confidence: float | None = None,
    intersect_with: str | sitk.Image = None,
    min_overlap: float = 0.1,
    input_file_name: str = "volume",
    output_file_name: str = "probabilities",
    class_idx: int | list[int] = 1,
) -> sitk.Image:
    """
    Exports a SITK probability mask. Applies a candidate extraction protocol
    (i.e. filtering probabilities above proba_threshold, applying connected
    component analysis and filtering out objects whose maximum probability is
    lower than min_confidence).

    Args:
        mask (sitk.Image): an SITK file object corresponding to a mask.
        metadata_path (str): path to metadata template file.
        proba_threshold (float, optional): sets values below this value to 0.
        min_confidence (float, optional): removes objects whose maximum
            probability is lower than this value.
        intersect_with (str | sitk.Image, optional): calculates the
            intersection of each candidate with the image specified in
            intersect_with. If the intersection is larger than
            min_intersection, the candidate is kept; otherwise it is discarded.
            Defaults to None.
        min_overlap (float, optional): minimum intersection over the union to keep
            candidate. Defaults to 0.1.
        input_file_name (str, optional): input file name. Defaults to "volume".
        output_file_name (str, optional): output file name. Defaults to
            "probabilities".
        class_idx (int | list[int], optional): class index for output probability.

    Returns:
        sitk.Image: returns the probability mask after the candidate extraction
            protocol.
    """

    input_proba_map = f"{output_dir}/{input_file_name}.npz"
    output_proba_map = f"{output_dir}/{output_file_name}.nii.gz"
    input_file = sitk.ReadImage(sitk_files[0])
    proba_array: np.ndarray = np.load(input_proba_map)["probabilities"]
    if isinstance(class_idx, int):
        proba_array = proba_array[class_idx]
    elif isinstance(class_idx, (list, tuple)):
        proba_array = proba_array[class_idx].sum(0)
    proba_array, _, _ = extract_lesion_candidates(
        proba_array,
        threshold=proba_threshold,
        min_confidence=min_confidence,
        intersect_with=intersect_with,
        min_overlap=min_overlap,
    )
    proba_map = sitk.GetImageFromArray(proba_array)
    proba_map.CopyInformation(input_file)
    threshold = sitk.ThresholdImageFilter()
    threshold.SetLower(float(proba_threshold))
    threshold.SetUpper(1.0)
    proba_map = threshold.Execute(proba_map)
    print(f"writing probability map to {output_proba_map}")
    sitk.WriteImage(proba_map, output_proba_map)

    return proba_map


def export_fractional_dicom_seg(
    proba_map: sitk.Image,
    metadata_path: str,
    file_paths: list[str],
    output_dir: str,
    output_file_name: str = "probabilities",
):
    """
    Exports a SITK image mask as a fractional DICOM segmentation object.

    Args:
        mask (sitk.Image): an SITK file object corresponding to a mask.
        metadata_path (str): path to metadata template file.
        file_paths (list[str]): list of DICOM file paths corresponding to the
            original series.
        output_dir (str): path to output directory.
        output_file_name (str, optional): output file name. Defaults to
            "probabilities".

    Returns:
        str: "success" if the process was successful, "empty mask" if the SITK
            mask contained no values.
    """
    from pydicom_seg_writers import FractionalWriter

    metadata_template = pydicom_seg.template.from_dcmqi_metainfo(
        metadata_path.strip()
    )
    writer = FractionalWriter(
        template=metadata_template,
        skip_empty_slices=True,
        skip_missing_segment=False,
    )

    if sitk.GetArrayFromImage(proba_map).sum() == 0:
        return "empty probability map"

    dcm = writer.write(proba_map, file_paths[0])
    output_dcm_path = f"{output_dir}/{output_file_name}.dcm"
    print(f"writing dicom output to {output_dcm_path}")
    dcm.save_as(output_dcm_path)

    return "success"


def calculate_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the intersection of the union between arrays a and b.

    Args:
        a (np.ndarray): array.
        b (np.ndarray): array.

    Returns:
        float: float value for the intersection over the union.
    """
    intersection = np.logical_and(a == 1, a == b).sum()
    union = a.sum() + b.sum() - intersection
    return intersection / union

def calculate_iou_a_over_b(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates how much of a overlaps with b.

    Args:
        a (np.ndarray): array.
        b (np.ndarray): array.

    Returns:
        float: float value for the intersection over the union.
    """
    intersection = np.logical_and(a == 1, a == b).sum()
    union = a.sum()
    return intersection / union


def extract_lesion_candidates(
    softmax: np.ndarray,
    threshold: float = 0.10,
    min_confidence: float = None,
    min_voxels_detection: int = 10,
    max_prob_round_decimals: int = 4,
    intersect_with: str |np.ndarray | sitk.Image = None,
    min_overlap: float = 0.1,
) -> tuple[np.ndarray, list[tuple[int, float]], np.ndarray]:
    """
    Lesion candidate protocol as implemented in [1]. Essentially:

        1. Clips probabilities to be above a threshold
        2. Detects connected components
        3. Filters based on candidate size
        4. Filters based on maximum probability value
        5. Returns the connected components

    [1] https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/9eef43d3a8fb0d0cb3cfca3f51fda91daa94f988/src/report_guided_annotation/extract_lesion_candidates.py#L17

    Args:
        softmax (np.ndarray): array with softmax probability values.
        threshold (float, optional): threshold below which values are set to 0.
            Defaults to 0.10.
        min_confidence (float, optional): minimum maximum probability value for
            each object after connected component analysis. Defaults to None
            (no filtering).
        min_voxels_detection (int, optional): minimum object size in voxels.
            Defaults to 10.
        max_prob_round_decimals (int, optional): maximum number of decimal
            places. Defaults to 4.
        intersect_with (str | sitk.Image, optional): calculates the
            intersection of each candidate with the image specified in
            intersect_with. If the intersection is larger than
            min_intersection, the candidate is kept; otherwise it is discarded.
            Defaults to None.
        min_overlap (float, optional): minimum intersection over the union to keep
            candidate. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, list[tuple[int, float]], np.ndarray]: the output
            probability map, a list of confidence values, and the connected
            components array as returned by ndimage.label.
    """
    all_hard_blobs = np.zeros_like(softmax)
    confidences = []
    clipped_softmax = softmax.copy()
    clipped_softmax[softmax < threshold] = 0
    blobs_index, num_blobs = ndimage.label(
        clipped_softmax, structure=np.ones((3, 3, 3))
    )
    if min_confidence is None:
        min_confidence = threshold

    if intersect_with is not None:
        if isinstance(intersect_with, str):
            intersect_with = sitk.ReadImage(intersect_with)
        if isinstance(intersect_with, sitk.Image):
            intersect_with = sitk.GetArrayFromImage(intersect_with)

    for idx in range(1, num_blobs + 1):
        hard_mask = np.zeros_like(blobs_index)
        hard_mask[blobs_index == idx] = 1

        hard_blob = hard_mask * clipped_softmax
        max_prob = np.max(hard_blob)

        if np.count_nonzero(hard_mask) <= min_voxels_detection:
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        elif max_prob < min_confidence:
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        if intersect_with is not None:
            iou = calculate_iou_a_over_b(hard_mask, intersect_with)
            if iou < min_overlap:
                blobs_index[hard_mask.astype(bool)] = 0
                continue

        if max_prob_round_decimals is not None:
            max_prob = np.round(max_prob, max_prob_round_decimals)
        hard_blob[hard_blob > 0] = clipped_softmax[hard_blob > 0]  # max_prob
        all_hard_blobs += hard_blob
        confidences.append((idx, max_prob))
    return all_hard_blobs, confidences, blobs_index
