import numpy as np
from typing import Tuple, List
from scipy.ndimage import zoom
import numpy.typing as npt
from skimage import measure


def _find_scaling_factor(input_voxel_size: List[float], output_voxel_size: List[float]) -> List[float]:
    return [i_size / o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size)]


def scale_resolution(data: npt.ArrayLike, scaling_factor: List[float], order: int = 0):
    """
    Change data scale data
    Args:
        data: Array like
        scaling_factor: List of float
        order: (int) scaling interpolation order

    Returns:
        Array like
    """
    if np.array_equal(scaling_factor, [1, 1, 1]):
        return data
    return zoom(data, scaling_factor, order=order)


def scale_to_voxel_size(data: npt.ArrayLike,
                        input_voxel_size: List[float],
                        output_voxel_size: List[float],
                        order: int = 0) -> npt.ArrayLike:
    """
    Change data resolution from input_voxel_resolution to output_voxel_resolution
    Args:
        data: Array like
        input_voxel_size: List of float
        output_voxel_size: List of float
        order: (int) scaling interpolation order

    Returns:
        Array like
    """
    scaling_factor = _find_scaling_factor(input_voxel_size, output_voxel_size)
    return scale_resolution(data, scaling_factor, order)


def inplace_normalize_01(data: npt.ArrayLike, eps: int = 1e-16) -> None:
    """
    normalize data between (0, 1)
    Args:
        data: Array like
        eps: (default) 1e-16

    Returns:
        Array like
    """
    data -= data.min()
    data /= (data.max() + eps)


def inplace_normalize_range(data: npt.ArrayLike, data_range: Tuple = (0, 1)) -> None:
    """
    normalize data between data_range = (a, b)
    data = (b - a) * data + a
    Args:
        data: Array like
        data_range: (default) (0, 1)

    Returns:
        Array like
    """

    data *= (data_range[1] - data_range[0])
    data += data_range[0]


def relabel_segmentation(segmentation):
    """
    Relabel contiguously a segmentation image, non-touching instances with same id will be relabeled differently.
    To be noted that measure.label is different from ndimage.label
    Example:
        x = np.zeros((5, 5))
        x[:2], x[2], x[3:]= 1, 2, 1
        print(x)
        print(ndimage.label(x)[0])
        print(measure.label(x))
        x = np.zeros((5, 5))
        x[:2], x[2], x[3:]= 1, 0, 1
        print(x)
        print(ndimage.label(x)[0])
        print(measure.label(x))
    """
    return measure.label(segmentation)