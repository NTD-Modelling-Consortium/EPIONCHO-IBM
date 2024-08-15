import pytest

from epioncho_ibm.tools import Data, flatten_and_sort


def test_flatten_and_sort_empty_input():
    assert [] == flatten_and_sort([])


def test_flatten_and_sort_single_empty_row():
    with pytest.raises(AttributeError):
        flatten_and_sort([[]])


def test_flatten_and_sort_single_data_multiple_keys():
    inputData: list[Data] = [
        {
            (2000, 0, 80, "test_measure"): 1,
            (2000, 0, 80, "test_measure_2"): 1,
            (2001, 0, 80, "test_measure"): 1,
        }
    ]
    outputData = flatten_and_sort(inputData)
    assert len(outputData) == 3
    assert outputData[0] == (2000, 0, 80, "test_measure", 1)
    assert outputData[1] == (2000, 0, 80, "test_measure_2", 1)
    assert outputData[2] == (2001, 0, 80, "test_measure", 1)


def test_flatten_and_sort_multiple_data_single_key():
    inputData: list[Data] = [
        {(2000, 0, 80, "test_measure"): 1, (2001, 0, 80, "test_measure"): 1},
        {(2000, 0, 80, "test_measure"): 2, (2001, 0, 80, "test_measure"): 2},
    ]
    outputData = flatten_and_sort(inputData)
    assert len(outputData) == 2
    assert outputData[0] == (2000, 0, 80, "test_measure", 1, 2)
    assert outputData[1] == (2001, 0, 80, "test_measure", 1, 2)


def test_flatten_and_sort_multiple_data_multiple_keys():
    inputData: list[Data] = [
        {
            (2000, 0, 80, "test_measure"): 1,
            (2000, 0, 80, "test_measure_2"): 1,
            (2001, 0, 80, "test_measure"): 1,
            (2001, 0, 80, "test_measure_2"): 4,
        },
        {
            (2000, 0, 80, "test_measure"): 2,
            (2000, 0, 80, "test_measure_2"): 3,
            (2001, 0, 80, "test_measure"): 2,
            (2001, 0, 80, "test_measure_2"): 7,
        },
    ]
    outputData = flatten_and_sort(inputData)
    assert len(outputData) == 4
    assert outputData[0] == (2000, 0, 80, "test_measure", 1, 2)
    assert outputData[1] == (2000, 0, 80, "test_measure_2", 1, 3)
    assert outputData[2] == (2001, 0, 80, "test_measure", 1, 2)
    assert outputData[3] == (2001, 0, 80, "test_measure_2", 4, 7)
