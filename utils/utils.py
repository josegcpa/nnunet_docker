import os
import numpy as np
import SimpleITK as sitk
import pydicom_seg
from glob import glob
from typing import List, Dict
from pydicom import dcmread

from typing import Sequence, Union


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


def mode(a: np.ndarray) -> Union[int, float]:
    """
    Calculates the mode of an array.

    Args:
        a (np.ndarray): a numpy array.

    Returns:
        (Union[int,float]): the mode of a.
    """
    u, c = np.unique(a, return_counts=True)
    return u[np.argmax(c)]


def get_origin(positions, z_axis=2):
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
    """Reads a DICOM file as SITK, using z-spacing to order slices in the
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
    for dcm_file in file_paths:
        f = dcmread(dcm_file)
        if (0x0020, 0x0037) in f:
            orientation = f[0x0020, 0x0037].value
        if (0x0020, 0x0032) in f:
            fs.append(f)
            good_file_paths.append(dcm_file)
    if orientation is None:
        return "No orientation available"
    position = np.array([x[0x0020, 0x0032].value for x in fs])
    rankings = np.array([x[0x0020, 0x0013].value for x in fs])

    segment_selection = get_contiguous_arr_idxs(position, rankings)
    if segment_selection is not None:
        fs = [x for x in fs if x[0x0020, 0x0013].value in segment_selection]
    position = np.array([x[0x0020, 0x0032].value for x in fs])
    rankings = np.array([x[0x0020, 0x0013].value for x in fs])
    if segment_selection is None:
        return "Positions are incorrect"
    orientation = list(map(float, orientation))
    orientation_sitk = dicom_orientation_to_sitk_direction(orientation)
    z_axis = 2
    real_position = np.matmul(
        position, np.array(orientation_sitk).reshape([3, 3])
    )
    z_position = np.sort(real_position[:, z_axis])
    z_spacing = np.median(np.diff(z_position))
    if np.isclose(z_spacing, 0) == True:
        return "Incorrect z-spacing information"
    pixel_spacing = [*f[0x0028, 0x0030].value, z_spacing]
    fs = sorted(fs, key=lambda x: float(x[0x0020, 0x0032].value[z_axis]))

    origin_sitk = get_origin(position, z_axis=z_axis)
    pixel_spacing_sitk = pixel_spacing
    try:
        pixel_data = np.stack(
            [f.pixel_array for f in fs],
        )
    except Exception:
        return "Pixel data may be corrupted"

    sitk_image = sitk.GetImageFromArray(pixel_data)
    sitk_image.SetDirection(orientation_sitk)
    sitk_image.SetOrigin(origin_sitk)
    sitk_image.SetSpacing(pixel_spacing_sitk)

    for k in metadata:
        sitk_image.SetMetaData(k, metadata[k])

    return sitk_image, good_file_paths


def get_study_uid(dicom_dir: List[str]) -> str:
    """Returns the study UID field from a random file in dicom_dir.

    Args:
        dicom_dir (str): directory with dicom (.dcm) files.

    Returns:
        str: string corresponding to study UID.
    """

    return dcmread(glob(f"{dicom_dir}/*dcm")[0])[(0x0020, 0x000D)].value


def sitk_mask_to_dicom_seg(
    mask: sitk.Image,
    metadata_path: str,
    series_paths: List[str],
    study_name: str,
    output_dir: str,
) -> str:
    metadata_template = pydicom_seg.template.from_dcmqi_metainfo(metadata_path)
    writer = pydicom_seg.MultiClassWriter(
        template=metadata_template,
        skip_empty_slices=True,
        skip_missing_segment=False,
    )

    dcm = writer.write(mask, glob(os.path.join(series_paths[0], "*")))
    output_dcm_path = f"{output_dir.strip()}/{study_name}.dcm"
    print(f"writing to {output_dcm_path}")
    dcm.save_as(output_dcm_path)
    return output_dcm_path
