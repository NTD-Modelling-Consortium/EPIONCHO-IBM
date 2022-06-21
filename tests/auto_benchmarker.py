import json
import math
import os
import time
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

from tests.definitions.utils import FlatDict, flatten_dict

# From here on we will only use benchmarker_test_func and StateStats

"""
This is a prototype auto benchmarker
"""


class BenchmarkArray(BaseModel):
    mean: float
    st_dev: float

    @classmethod
    def from_array(cls, array: Union[NDArray[np.float_], NDArray[np.int_]]):
        return cls(mean=np.mean(array), st_dev=np.std(array))


class BaseSettingsModel(BaseModel):
    max_product: float
    benchmark_iters: int


class BaseOutputData(BaseModel):
    data: Dict[str, BenchmarkArray]


DataT = TypeVar("DataT")


class BaseTestDimension(GenericModel, Generic[DataT]):
    minimum: DataT
    maximum: DataT
    steps: int


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


FuncReturn = TypeVar(
    "FuncReturn",
    bound=BaseModel,
)


class AutoBenchmarker(Generic[FuncReturn]):
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
        self.pytest_config = PytestConfig.parse_file("pytest_config.json")
        self._settings_model = None
        self._output_data = None
        self.est_base_time = est_base_time

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

    def _generate_settings_file(self, settings_path: Path) -> None:
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

        settings_file = open(settings_path, "w+")
        json.dump(settings.dict(), settings_file, indent=2)

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

    def generate_benchmark(self, verbose: bool = False):
        SettingsModel = self.settings_model
        OutputModel = self.output_model
        TestData = create_model(
            "TestData", tests=(List[OutputModel], ...)  # type:ignore
        )

        settings_folder = Path(self.pytest_config.benchmark_path)
        settings_path = Path(str(settings_folder) + os.sep + "settings.json")
        if not settings_path.exists():
            if not settings_folder.exists():
                settings_folder.mkdir(parents=True)
            self._generate_settings_file(settings_path)
        settings_model = SettingsModel.parse_file(settings_path)
        test_pairs, total_product = get_test_pairs(
            settings=settings_model, parameters=self.parameters
        )
        if verbose:
            est_test_time = self.est_base_time * total_product
            est_benchmark_time = est_test_time * settings_model.benchmark_iters
            print(f"Benchmark will run {len(test_pairs)} tests")
            print(f"Estimated benchmark calc time (one core): {est_benchmark_time}")
            print(
                f"Estimated benchmark calc time (multiple cores): {est_benchmark_time/cpu_count()}"
            )
            print(f"Estimated test time (no reruns): {est_test_time}")

        start = time.time()
        tests: List[BaseOutputData] = []
        headers = [k for k in self.parameters.keys()]
        for items in test_pairs:
            list_of_stats: List[FuncReturn] = Parallel(n_jobs=cpu_count())(
                delayed(self.func)(*items)
                for _ in range(settings_model.benchmark_iters)
            )
            data = self._compute_mean_and_st_dev_of_pydantic(list_of_stats)
            values = {header: items[i] for i, header in enumerate(headers)}

            test_output = OutputModel(data=data, **values)
            tests.append(test_output)
        end = time.time()
        if verbose:
            print(f"Benchmark calculated in: {end-start}")
        test_data = TestData(tests=tests)
        benchmark_file_path = Path(str(settings_folder) + os.sep + "benchmark.json")
        benchmark_file = open(benchmark_file_path, "w+")
        json.dump(test_data.dict(), benchmark_file, indent=2)


from epioncho_ibm import Params, RandomConfig, State, run_simulation
from epioncho_ibm.state import StateStats


def benchmarker_test_func(end_time: float, population: int) -> StateStats:
    params = Params(human_population=population)
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.to_stats()
    return stats


benchmarker = AutoBenchmarker[StateStats](benchmarker_test_func)
mod = benchmarker.generate_benchmark(verbose=True)
