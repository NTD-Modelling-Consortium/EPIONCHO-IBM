import dis
import io
import json
import math
import os
import time
from contextlib import redirect_stdout
from inspect import signature
from multiprocessing import cpu_count
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel, create_model
from pydantic.generics import GenericModel

from auto_tests.definitions.utils import FlatDict, flatten_dict

# From here on we will only use benchmarker_test_func and StateStats

"""
This is a prototype auto benchmarker

TODO: Future feature, support enums in functions and iterate over each possibility

Note - eventually make autobenchmarker and funcbenchmarker inherit from benchmarker
"""


class BenchmarkArray(BaseModel):
    mean: float
    st_dev: float

    @classmethod
    def from_array(cls, array: Union[NDArray[np.float_], NDArray[np.int_]]):
        return cls(mean=np.mean(array), st_dev=np.std(array))


class BaseSettingsModel(BaseModel):
    """
    The settings corresponding to one function
    """

    max_product: float
    benchmark_iters: int


class GlobalSettingsModel(BaseModel):
    """
    The settings corresponding to all functions
    """


class BaseOutputData(BaseModel):
    data: Dict[str, BenchmarkArray]


DataT = TypeVar("DataT")


class BaseTestDimension(GenericModel, Generic[DataT]):
    minimum: DataT
    maximum: DataT
    steps: int


class BaseTestModel(BaseModel):
    pass


def get_test_pairs(
    settings: BaseSettingsModel, parameters: Dict[str, type]
) -> Tuple[List[Tuple], float]:
    def get_exponentially_spaced_steps(
        start: Union[int, float], end: Union[int, float], n_steps: int
    ) -> NDArray[np.float_]:
        log_start = math.log(start)
        log_end = math.log(end)
        exp_spaced = np.linspace(log_start, log_end, n_steps)
        return np.exp(exp_spaced)

    arrays: List[Union[NDArray[np.int_], NDArray[np.float_]]] = []
    for k, t in parameters.items():
        setting_item: BaseTestDimension = getattr(settings, k)
        if issubclass(t, int):
            spaces: Union[NDArray[np.int_], NDArray[np.float_]] = np.round(
                get_exponentially_spaced_steps(
                    setting_item.minimum, setting_item.maximum, setting_item.steps
                )
            )
        else:
            spaces: Union[
                NDArray[np.int_], NDArray[np.float_]
            ] = get_exponentially_spaced_steps(
                setting_item.minimum, setting_item.maximum, setting_item.steps
            )
        arrays.append(spaces)
    new_arrays = []
    no_arrays = len(arrays)
    for i, array in enumerate(arrays):
        ones = [1] * no_arrays
        ones[i] = len(array)
        new_arrays.append(np.reshape(array, tuple(ones)))
    combined_array = np.multiply(*new_arrays)
    valid_tests = combined_array < settings.max_product
    coords = valid_tests.nonzero()
    items_for_test = []
    for i, spaces in enumerate(arrays):
        item_for_test = spaces[coords[i]]
        items_for_test.append(item_for_test)
    total_product = np.sum(combined_array[valid_tests])
    return list(zip(*tuple(items_for_test))), total_product


def snake_to_camel_case(string: str) -> str:
    string_elems = string.split("_")
    new_elems = [i.title() for i in string_elems]
    return "".join(new_elems)


class PytestConfig(BaseModel):
    acceptable_st_devs: float
    re_runs: int
    benchmark_path: str


def get_func_info(func) -> str:
    with redirect_stdout(io.StringIO()) as f:
        dis.dis(func)
    return f.getvalue()


FuncReturn = TypeVar(
    "FuncReturn",
    bound=BaseModel,
)


class SetupFuncBenchmarker(Generic[FuncReturn]):
    parameters: Dict[str, type]
    func: Callable[..., FuncReturn]
    return_type: FuncReturn
    func_name: str
    camel_name: str
    pytest_config: PytestConfig
    _settings_model: Optional[Type[BaseSettingsModel]]
    _output_data: Optional[Type[BaseOutputData]]

    def __init__(
        self, func: Callable[..., FuncReturn], est_base_time: float = 0.46
    ) -> None:
        sig = signature(func)
        return_type: FuncReturn = sig.return_annotation
        if not issubclass(
            return_type, BaseModel  # type:ignore BaseModel incompatible with *?
        ):
            raise ValueError("Function return must inherit from BaseModel")
        parameters = {v.name: v.annotation for _, v in sig.parameters.items()}
        self.parameters = parameters
        self.func = func
        self.return_type = return_type
        self.func_name = func.__name__
        self.camel_name = snake_to_camel_case(self.func_name)

        self._settings_model = None
        self._output_data = None
        self._test_model = None
        self.est_base_time = est_base_time
        self.func_info = (
            get_func_info(self.func)
            + str(self.parameters)
            + str(self.return_type.schema())
        )

    def __str__(self) -> str:
        return self.func_info

    def _generate_settings_model(self) -> Type[BaseSettingsModel]:
        attributes: Dict[str, Tuple[type, ellipsis]] = {
            k: (BaseTestDimension[t], ...)  # type:ignore
            for k, t in self.parameters.items()
        }
        return create_model(
            self.camel_name + "Settings", **attributes, __base__=BaseSettingsModel
        )

    @property
    def settings_model(self) -> Type[BaseSettingsModel]:
        if self._settings_model is None:
            self._settings_model = self._generate_settings_model()
        return self._settings_model

    def generate_settings_instance(self) -> BaseSettingsModel:
        model = self.settings_model
        attr_dict = {}
        for k, t in self.parameters.items():
            print(f"Attributes for {k}: \n")
            while True:
                constraints = input(
                    f"Enter (minimum: {str(t.__name__)}, maximum: {str(t.__name__)}, steps: int): "
                )
                items = constraints.split(",")
                if len(items) == 3:
                    try:
                        minimum = t(items[0])
                        maximum = t(items[1])
                        steps = int(items[2])
                        if maximum < minimum:
                            print("max less than min")
                            continue
                        break
                    except ValueError:
                        print("invalid type")
                        continue
                else:
                    print("incorrect number of args")
                    continue
            attr_dict[k] = BaseTestDimension[t](  # type:ignore
                minimum=minimum, maximum=maximum, steps=steps
            )
        while True:
            try:
                max_product_string = input("max_product: float: ")
                max_product = float(max_product_string)
                break
            except ValueError:
                print("invalid type")
                continue
        attr_dict["max_product"] = max_product

        while True:
            try:
                benchmark_iters_string = input("benchmark_iters: int: ")
                benchmark_iters = int(benchmark_iters_string)
                break
            except ValueError:
                print("invalid type")
                continue
        attr_dict["benchmark_iters"] = benchmark_iters

        settings = model.parse_obj(attr_dict)
        return settings

    def _generate_output_model(self) -> Type[BaseOutputData]:
        attributes: Dict[str, Tuple[type, ellipsis]] = {
            k: (t, ...) for k, t in self.parameters.items()
        }
        return create_model(
            self.camel_name + "Settings", **attributes, __base__=BaseOutputData
        )

    @property
    def output_model(self) -> Type[BaseOutputData]:
        if self._output_data is None:
            self._output_data = self._generate_output_model()
        return self._output_data


SettingsModel = TypeVar("SettingsModel", bound=BaseSettingsModel)
FuncReturn = TypeVar("FuncReturn", bound=BaseModel)


class FuncBenchmarker(Generic[SettingsModel, FuncReturn]):
    def __init__(
        self, settings: SettingsModel, func_setup: SetupFuncBenchmarker
    ) -> None:
        self.settings = settings
        self.func_setup = func_setup
        self.test_pairs, self.total_product = get_test_pairs(
            settings, func_setup.parameters
        )

    def __len__(self) -> int:
        return len(self.test_pairs)

    def estimate_computation_time(self) -> Tuple[float, float]:
        est_test_time = self.func_setup.est_base_time * self.total_product
        est_benchmark_time = est_test_time * self.settings.benchmark_iters
        return est_test_time, est_benchmark_time

    def _compute_mean_and_st_dev_of_pydantic(
        self,
        input_stats: List[FuncReturn],
    ) -> Dict[str, BenchmarkArray]:
        flat_dicts: List[FlatDict] = [
            flatten_dict(input_stat.dict()) for input_stat in input_stats
        ]
        dict_of_arrays: Dict[str, List[Any]] = {}
        for flat_dict in flat_dicts:
            for k, v in flat_dict.items():
                if k in dict_of_arrays:
                    dict_of_arrays[k].append(v)
                else:
                    dict_of_arrays[k] = [v]
        final_dict_of_arrays: Dict[str, NDArray[np.float_]] = {
            k: np.array(v) for k, v in dict_of_arrays.items()
        }
        return {
            k: BenchmarkArray.from_array(v) for k, v in final_dict_of_arrays.items()
        }

    def generate_benchmark(self) -> List[BaseOutputData]:
        OutputModel = self.func_setup.output_model

        tests: List[BaseOutputData] = []
        headers = [k for k in self.func_setup.parameters.keys()]
        for items in self.test_pairs:
            list_of_stats: List[FuncReturn] = Parallel(n_jobs=cpu_count())(
                delayed(self.func_setup.func)(*items)
                for _ in range(self.settings.benchmark_iters)
            )
            data = self._compute_mean_and_st_dev_of_pydantic(list_of_stats)
            values = {header: items[i] for i, header in enumerate(headers)}

            test_output = OutputModel(data=data, **values)
            tests.append(test_output)
        return tests

    def test_benchmark_data(
        self, benchmark_data: BaseOutputData, acceptable_st_devs: float
    ) -> None:
        func_args = {
            dimension: getattr(benchmark_data, dimension)
            for dimension in self.func_setup.parameters.keys()
        }
        func_return = self.func_setup.func(**func_args)
        func_return_dict = flatten_dict(func_return.dict())
        for k, v in func_return_dict.items():
            if k not in benchmark_data.data:
                raise RuntimeError(f"Key {k} not present in benchmark")
            else:
                benchmark_item = benchmark_data.data[k]
            benchmark_item_mean = benchmark_item.mean
            benchmark_item_st_dev = benchmark_item.st_dev
            benchmark_lower_bound = (
                benchmark_item_mean - acceptable_st_devs * benchmark_item_st_dev
            )
            benchmark_upper_bound = (
                benchmark_item_mean + acceptable_st_devs * benchmark_item_st_dev
            )
            if v < benchmark_lower_bound:
                raise ValueError(
                    f"For key: {k} lower bound: {benchmark_lower_bound} surpassed by value {v}"
                )
            if v > benchmark_upper_bound:
                raise ValueError(
                    f"For key: {k} upper bound: {benchmark_upper_bound} surpassed by value {v}"
                )


class AutoBenchmarker:
    def __init__(self, **funcs: Callable) -> None:
        self.pytest_config = PytestConfig.parse_file("pytest_config.json")
        self.setup_func_benchmarkers = {
            k: SetupFuncBenchmarker(v) for k, v in funcs.items()
        }
        self._settings_model = None
        self._settings = None
        self._test_model = None
        self._func_benchmarkers = None
        self.settings_folder = Path(self.pytest_config.benchmark_path)

    def __str__(self) -> str:
        return ", ".join(
            [name + str(i) for name, i in self.setup_func_benchmarkers.items()]
        )

    def _generate_test_model(self) -> Type[BaseTestModel]:
        output_models_dict = {
            func_name: (List[setup_func_benchmarker.output_model], ...)  # type:ignore
            for func_name, setup_func_benchmarker in self.setup_func_benchmarkers.items()
        }
        return create_model(
            "TestData",
            __base__=BaseTestModel,
            tests=(create_model("FuncData", **output_models_dict), ...),  # type:ignore
        )

    @property
    def test_model(self) -> Type[BaseTestModel]:
        if self._test_model is None:
            self._test_model = self._generate_test_model()
        return self._test_model

    def _generate_settings_model(self) -> Type[GlobalSettingsModel]:
        settings_dict = {}
        for func_name, func_benchmarker in self.setup_func_benchmarkers.items():
            settings_dict[func_name] = (func_benchmarker.settings_model, ...)
        return create_model(
            "GlobalSettings", __base__=GlobalSettingsModel, **settings_dict
        )

    @property
    def settings_model(self) -> Type[GlobalSettingsModel]:
        if self._settings_model is None:
            self._settings_model = self._generate_settings_model()
        return self._settings_model

    def _generate_settings_file(self, settings_path: Path) -> None:
        model = self.settings_model
        settings = model.parse_obj(
            {
                k: v.generate_settings_instance()
                for k, v in self.setup_func_benchmarkers.items()
            }
        )
        settings_file = open(settings_path, "w+")
        json.dump(settings.dict(), settings_file, indent=2)

    @property
    def settings(self) -> GlobalSettingsModel:
        if self._settings is None:
            settings_path = Path(str(self.settings_folder) + os.sep + "settings.json")
            if not settings_path.exists():
                if not self.settings_folder.exists():
                    self.settings_folder.mkdir(parents=True)
                self._generate_settings_file(settings_path)
            self._settings = self.settings_model.parse_file(settings_path)
        return self._settings

    @property
    def func_benchmarkers(self) -> Dict[str, FuncBenchmarker]:
        if self._func_benchmarkers is None:
            self._func_benchmarkers = {
                func_name: FuncBenchmarker(
                    getattr(self.settings, func_name), func_setup
                )
                for func_name, func_setup in self.setup_func_benchmarkers.items()
            }
        return self._func_benchmarkers

    def generate_benchmark(self, verbose: bool = False):
        func_benchmarkers = self.func_benchmarkers

        if verbose:
            total_benchmark_time = 0
            test_times = []
            total_tests = 0
            for func_benchmarker in func_benchmarkers.values():
                (
                    est_test_time,
                    est_benchmark_time,
                ) = func_benchmarker.estimate_computation_time()
                total_benchmark_time += est_benchmark_time
                total_tests += len(func_benchmarker)
                test_times.append(est_test_time)
            total_test_time = sum(test_times)
            print(f"Benchmark will run {total_tests} tests")
            print(f"Estimated benchmark calc time (one core): {total_benchmark_time}")
            print(
                f"Estimated benchmark calc time (multiple cores): {total_benchmark_time/cpu_count()}"
            )
            print(f"Estimated total test time (no reruns): {total_test_time}")

        start = time.time()
        all_benchmarks_out = {}
        for func_name, func_benchmarker in func_benchmarkers.items():
            benchmark_out = func_benchmarker.generate_benchmark()
            all_benchmarks_out[func_name] = benchmark_out
        end = time.time()

        if verbose:
            print(f"Benchmark calculated in: {end-start}")
        test_data = self.test_model.parse_obj({"tests": all_benchmarks_out})

        benchmark_file_path = Path(
            str(self.settings_folder) + os.sep + "benchmark.json"
        )
        benchmark_file = open(benchmark_file_path, "w+")
        json.dump(test_data.dict(), benchmark_file, indent=2)
        benchmark_file.close()
        hash_file_path = Path(str(self.settings_folder) + os.sep + "data_hash.txt")
        with open(hash_file_path, "w+") as f:
            f.write(str(self))

    def test_benchmark_data(
        self, benchmark_data: BaseOutputData, acceptable_st_devs: float, func_name: str
    ) -> None:
        func_benchmarker = self.func_benchmarkers[func_name]
        func_benchmarker.test_benchmark_data(benchmark_data, acceptable_st_devs)
