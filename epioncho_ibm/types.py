import numpy as np
from numpy.typing import NDArray


class _SubType:
    Float = NDArray[np.float_]
    Int = NDArray[np.int_]
    Bool = NDArray[np.bool_]


class _SubAxis(_SubType):
    General = _SubType
    Person = _SubType


class Array(_SubAxis):
    """
    Typing for numpy arrays.

    Every combination is a representation in one of these ways:

    Array.Type
    Array.1stAxis.Type
    Array.1stAxis.2ndAxis.Type

    Possible Types:
    Float, Int, Bool

    Possible 1st Axis - axis has length:
    Person: Number of people
    WormCat: Number of Worm Categories
    MFCat: Number of MF  Categories
    WormDelay: Worm Delay Columns (Time steps of delay)
    MFDelay: MF Delay Columns (Time steps of delay)
    ExposureDelay: Exposure Delay Columns (Time steps of delay)
    L1Delay: L1 Delay Columns (Time steps of delay)
    Treatments: Number of treatments
    General: Any length

    Possible 2nd Axis:
    Person: as above
    General: as above
    """

    WormCat = _SubAxis
    MFCat = _SubAxis
    WormDelay = _SubAxis
    MFDelay = _SubAxis
    ExposureDelay = _SubAxis
    L1Delay = _SubAxis
    Treatments = _SubAxis
