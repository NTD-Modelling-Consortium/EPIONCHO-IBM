import numpy as np
from numpy.typing import NDArray


class _SubType:
    Float = NDArray[np.float_]
    Int = NDArray[np.int_]
    Bool = NDArray[np.bool_]


class _SubCat(_SubType):
    WormCat = _SubType
    MFCat = _SubType
    WormDelay = _SubType
    MFDelay = _SubType
    ExposureDelay = _SubType
    L1Delay = _SubType
    General = _SubType
    Treatments = _SubType


class Array(_SubCat):
    """
    Typing for numpy arrays.


    ### Options:
    * `Array.General`: 1st Axis is non specific
    * `Array.Person`: 1st Axis represents Person

    """

    Person = _SubCat
