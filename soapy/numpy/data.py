import numpy as np
import numpy.typing as npt
import yaml
from typing import Tuple, List
from dataclasses import dataclass, field
from plantseg.utils.utils import scale_resolution, scale_to_voxel_size
from plantseg.utils.utils import inplace_normalize_range, inplace_normalize_01
from plantseg.utils.utils import relabel_segmentation
from typing import Protocol
import copy


class ImageStack(np.ndarray):
    """
    Container for image like data
    """
    element_size_um: Tuple[float]
    original_element_sizedtype_um: Tuple[float]
    meta: dict
    ndim: int
    data_range: Tuple[float]
    interpolation_order: int
    dtype: str

    def __new__(cls, np_array,
                stack_ndim: int,
                element_size_um: Tuple[float] = None,
                meta: dict = None,
                data_range: Tuple[float] = (0, 1),
                interpolation_order: int = 1):
        obj = np.asarray(np_array).view(cls)
        # add the new attribute to the created instance
        obj.stack_ndim = 0
        obj._element_size_um = element_size_um if element_size_um is not None else (1, 1, 1)
        obj.meta = meta if meta is not None else {}
        obj.data_range = data_range
        obj.interpolation_order = interpolation_order

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        for key in ['meta',
                    '_element_size_um',
                    'stack_ndim',
                    'data_range',
                    'interpolation_order']:
            _attr = getattr(obj, key, None)
            setattr(self, key, _attr)

    @property
    def element_size_um(self):
        return self._element_size_um

    def _to_int_type(self, data_type: str = 'uint8'):
        self.data_range = (0, np.iinfo(data_type).max)
        self.normalize()
        self.dtype = data_type

    def to_uint8(self):
        self._to_int_type('uint8')

    def to_uint16(self):
        self._to_int_type('uint16')

    def normalize_01(self):
        # cast to float64 to normalize
        self.dtype = 'float64'
        inplace_normalize_01(self)

    def normalize(self):
        # ensure data is between 0-1
        self.normalize_01()
        # normalize it between data_range
        inplace_normalize_range(self, self.data_range)


@dataclass
class SegmentationStack:
    """
    Container for segmentation like data
    """
    data: npt.ArrayLike = field(repr=False)
    shape: Tuple[int] = field(default=None, repr=True)
    element_size_um: Tuple[float] = field(default=None, repr=True)
    meta: dict = field(default=None, repr=False)
    ndim: int = field(default=3, repr=True)
    data_type: str = field(default='uint32', repr=True)
    order: int = field(default=2, repr=False)

    def __post_init__(self):
        self.check()

    def check(self):
        self.data = self.data.astype(self.data_type)

    def _to_int_type(self, data_type: str = 'uint8'):
        self.data_range = (0, np.iinfo(data_type).max)

        if self.data.max() > self.data_range[1]:
            self.relabel()
            assert self.data.max() < self.data_range[1]

        self.data_type = data_type
        self.data = self.data.astype(self.data_type)

    def to_uint16(self):
        self._to_int_type('uint16')

    def relabel(self):
        self.data = relabel_segmentation(self.data)