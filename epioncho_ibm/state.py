from pathlib import Path
from typing import IO, Callable, Generic, TypeVar

import h5py
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from epioncho_ibm.treatment import get_treatment

from .blackfly import calc_l1, calc_l2, calc_l3, calc_new_worms_from_blackfly
from .derived_params import DerivedParams
from .exposure import calculate_total_exposure
from .microfil import calculate_microfil_delta
from .params import Params
from .people import People
from .types import Array
from .worms import WormGroup, calculate_new_worms

np.seterr(all="ignore")


class NumericArrayStat(BaseModel):
    mean: float
    # st_dev: float

    @classmethod
    def from_array(cls, array: Array.Person.Float | Array.Person.Int):
        return cls(mean=float(np.mean(array)))  # , st_dev=np.std(array))


class StateStats(BaseModel):
    percent_compliant: float
    percent_male: float
    L1: NumericArrayStat
    L2: NumericArrayStat
    L3: NumericArrayStat
    ages: NumericArrayStat
    mf: NumericArrayStat
    male_worms: NumericArrayStat
    infertile_female_worms: NumericArrayStat
    fertile_female_worms: NumericArrayStat
    mf_per_skin_snip: float
    population_prevalence: float


def negative_binomial_alt_interface(
    n: Array.General.Float, mu: Array.General.Float
) -> Array.General.Int:
    """
    Provides an alternate interface for random negative binomial.

    Args:
        n (Array.General.Float): Number of successes
        mu (Array.General.Float): Mean of the distribution

    Returns:
        Array.General.Int: Samples from a negative binomial distribution
    """
    non_zero_n = n[n > 0]
    rel_prob = non_zero_n / (non_zero_n + mu[n > 0])
    temp_output = np.random.negative_binomial(
        n=non_zero_n, p=rel_prob, size=len(non_zero_n)
    )
    output = np.zeros(len(n), dtype=int)
    output[n > 0] = temp_output
    return output


CallbackStat = TypeVar("CallbackStat")


class State(Generic[CallbackStat]):
    _people: People
    _params: Params
    _derived_params: DerivedParams

    def __init__(self, people: People, params: Params) -> None:
        """
        The state of the model at any one time.

        Args:
            people (People): A representation of the People in the model.
            params (Params): A set of fixed parameters for controlling the model.
        """
        self._people = people
        self.params = params

    @classmethod
    def from_params(
        cls, params: Params, n_people: int, gamma_distribution: float = 0.3
    ):
        """
        Generate the initial state of the model, based on parameters.

        Args:
            params (Params): A set of fixed parameters for controlling the model.
            n_people (int): The number of people to be simulated
            gamma_distribution (float, optional): Individual level exposure heterogeneity. Defaults to 0.3.

        Returns:
            State: The state of the model
        """
        return cls(
            people=People.from_params(params, n_people, gamma_distribution),
            params=params,
        )

    @property
    def params(self):
        """
        A set of fixed parameters for controlling the model.
        """
        return self._params

    @params.setter
    def params(self, value: object):
        assert isinstance(value, Params)
        self._derived_params = DerivedParams(value)
        self._params = value

    @property
    def n_people(self):
        """
        The number of people simulated.
        """
        return len(self._people)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, State)
            and self._people == other._people
            and self.params == other.params
        )

    def _advance(self, current_time: float, debug: bool):
        """
        Advance the state forward one time step from t to t + dt

        Args:
            current_time (float): The current time (t) in the model.
        """

        treatment = get_treatment(
            self.params.treatment,
            self.params.humans,
            self.params.delta_time,
            current_time,
            self._derived_params.treatment_times,
            self._people.ages,
            self._people.compliance,
        )

        total_exposure = calculate_total_exposure(
            self.params.exposure,
            self._people.ages,
            self._people.sex_is_male,
            self._people.individual_exposure,
        )
        self._people.ages += self.params.delta_time

        old_worms = self._people.worms.copy()

        # there is a delay in new parasites entering humans (from fly bites) and
        # entering the first adult worm age class
        new_worms = calc_new_worms_from_blackfly(
            self._people.blackfly.L3,
            self.params.blackfly,
            self.params.delta_time,
            total_exposure,
            self.n_people,
            old_worms,
            debug,
        )

        if self._people.delay_arrays.current_worm_delay is None:
            worm_delay: Array.Person.Int = new_worms
        else:
            worm_delay: Array.Person.Int = self._people.delay_arrays.current_worm_delay

        self._people.worms, last_time_of_last_treatment = calculate_new_worms(
            current_worms=self._people.worms,
            worm_params=self.params.worms,
            treatment=treatment,
            time_of_last_treatment=self._people.time_of_last_treatment,
            delta_time=self.params.delta_time,
            worm_delay_array=worm_delay,
            mortalities=self._derived_params.worm_mortality_rate,
            mortalities_generator=self._derived_params.worm_mortality_generator,
            current_time=current_time,
            debug=debug,
            worm_age_rate_generator=self._derived_params.worm_age_rate_generator,
            worm_sex_ratio_generator=self._derived_params.worm_sex_ratio_generator,
            worm_lambda_zero_generator=self._derived_params.worm_lambda_zero_generator,
            worm_omega_generator=self._derived_params.worm_omega_generator,
        )

        if (
            self.params.treatment is not None
            and current_time >= self.params.treatment.start_time
        ):
            self._people.time_of_last_treatment = last_time_of_last_treatment

        # inputs for delay in L1

        old_mf: Array.Person.Float = np.sum(self._people.mf, axis=0)
        self._people.mf += calculate_microfil_delta(
            current_microfil=self._people.mf,
            delta_time=self.params.delta_time,
            microfil_params=self.params.microfil,
            treatment_params=self.params.treatment,
            microfillarie_mortality_rate=self._derived_params.microfillarie_mortality_rate,
            fecundity_rates_worms=self._derived_params.fecundity_rates_worms,
            time_of_last_treatment=self._people.time_of_last_treatment,
            current_time=current_time,
            current_fertile_female_worms=old_worms.fertile,
            current_male_worms=old_worms.male,
            debug=debug,
        )
        old_blackfly_L1 = self._people.blackfly.L1

        if self._people.delay_arrays.exposure_delay is None:
            exposure_delay: Array.Person.Float = total_exposure
        else:
            exposure_delay: Array.Person.Float = (
                self._people.delay_arrays.exposure_delay
            )

        if self._people.delay_arrays.current_mf_delay is None:
            mf_delay: Array.Person.Float = old_mf
        else:
            mf_delay: Array.Person.Float = self._people.delay_arrays.current_mf_delay

        self._people.blackfly.L1 = calc_l1(
            self.params.blackfly,
            old_mf,
            mf_delay,
            total_exposure,
            exposure_delay,
            self.params.year_length_days,
        )

        old_blackfly_L2 = self._people.blackfly.L2
        self._people.blackfly.L2 = calc_l2(
            self.params.blackfly,
            old_blackfly_L1,
            mf_delay,
            exposure_delay,
            self.params.year_length_days,
        )
        self._people.blackfly.L3 = calc_l3(self.params.blackfly, old_blackfly_L2)
        # TODO: Resolve new_mf=old_mf
        self._people.delay_arrays.lag_all_arrays(
            new_worms=new_worms, total_exposure=total_exposure, new_mf=old_mf
        )
        people_to_die: Array.Person.Bool = np.logical_or(
            self._derived_params.people_to_die_generator.binomial(
                np.repeat(1, self.n_people)
            )
            == 1,
            self._people.ages >= self.params.humans.max_human_age,
        )
        self._people.process_deaths(people_to_die, self.params.humans.gender_ratio)

    def run_simulation(
        self,
        *,
        start_time: float = 0,
        end_time: float,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Run the simulation between two times.

        Args:
            start_time (float, optional): The time (in years) to start the simulation. Defaults to 0.
            end_time (float): The time (in years) to end the simulation. Defaults to 0.
            verbose (bool, optional): When true, the model displays a progress bar. Defaults to False.
            debug (bool, optional): When true, the model makes additional validations on the data. Defaults to False.
        Raises:
            ValueError: End time after start
        """
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        # total progress bar must be a bit over so that the loop doesn't exceed total
        with tqdm(
            total=end_time - start_time + self.params.delta_time, disable=not verbose
        ) as progress_bar:
            while current_time < end_time:
                progress_bar.update(self.params.delta_time)
                self._advance(current_time=current_time, debug=debug)
                current_time += self.params.delta_time

    def run_simulation_output_stats(
        self,
        sampling_interval: float,
        start_time: float = 0,
        end_time: float = 0,
        verbose: bool = False,
        debug: bool = False,
    ) -> list[tuple[float, StateStats]]:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        output_stats: list[tuple[float, StateStats]] = []
        while current_time < end_time:
            if self.params.delta_time > current_time % 0.2 and verbose:
                print(current_time)
            if self.params.delta_time > current_time % sampling_interval:
                output_stats.append((current_time, self.to_stats()))
            self._advance(current_time=current_time, debug=debug)
            current_time += self.params.delta_time
        return output_stats

    def run_simulation_output_callback(
        self,
        output_callback: Callable[[People, float], CallbackStat],
        sampling_interval: float,
        start_time: float = 0,
        end_time: float = 0,
        verbose: bool = False,
        debug: bool = False,
    ) -> list[CallbackStat]:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        output_stats: list[CallbackStat] = []
        while current_time < end_time:
            if self.params.delta_time > current_time % 0.2 and verbose:
                print(current_time)
            if self.params.delta_time > current_time % sampling_interval:
                output_stats.append(output_callback(self._people, current_time))
            self._advance(current_time=current_time, debug=debug)
            current_time += self.params.delta_time
        return output_stats

    @classmethod
    def from_hdf5(cls, input_file: str | Path | IO[bytes]):
        f = h5py.File(input_file, "r")
        people_group = f["people"]
        assert isinstance(people_group, h5py.Group)
        params: str = str(f.attrs["params"])
        return cls(People.from_hdf5(people_group), Params.parse_raw(params))

    def to_hdf5(self, output_file: str | Path | IO[bytes]):
        f = h5py.File(output_file, "w")
        group_people = f.create_group("people")
        self._people.to_hdf5(group_people)
        f.attrs["params"] = self._params.json()

    def microfilariae_per_skin_snip(
        self,
    ) -> tuple[float, Array.Person.Float]:
        """
        Calculates number of mf in skin snip for all people.

        People are tested for the presence of mf using a skin snip.
        We assume mf are overdispersed in the skin

        Returns:
            tuple[float, Array.Person.Float]: Mean mf, mf by person.
        """
        kmf = (
            self.params.microfil.slope_kmf
            * np.sum(
                self._people.worms.fertile + self._people.worms.infertile,
                axis=0,
            )
            + self.params.microfil.initial_kmf
        )

        mu = self.params.humans.skin_snip_weight * np.sum(self._people.mf, axis=0)
        if self.params.humans.skin_snip_number > 1:
            total_skin_snip_mf = np.zeros(
                (
                    self.n_people,
                    self.params.humans.skin_snip_number,
                )
            )
            for i in range(self.params.humans.skin_snip_number):
                total_skin_snip_mf[:, i] = negative_binomial_alt_interface(n=kmf, mu=mu)
            mfobs: Array.Person.Int = np.sum(total_skin_snip_mf, axis=1)
        else:
            mfobs: Array.Person.Int = negative_binomial_alt_interface(n=kmf, mu=mu)

        mfobs_percent: Array.Person.Float = mfobs / (
            self.params.humans.skin_snip_number * self.params.humans.skin_snip_weight
        )
        return float(np.mean(mfobs_percent)), mfobs_percent

    def mf_prevalence_in_population(self) -> float:
        """
        Calculates mf prevalence in population.

        Returns:
            float: mf_prevalence
        """
        pop_over_min_age_array = (
            self._people.ages >= self.params.humans.min_skinsnip_age
        )
        _, mf_skin_snip = self.microfilariae_per_skin_snip()
        infected_over_min_age = float(np.sum(mf_skin_snip[pop_over_min_age_array] > 0))
        total_over_min_age = float(np.sum(pop_over_min_age_array))
        return infected_over_min_age / total_over_min_age

    def to_stats(self) -> StateStats:
        return StateStats(
            percent_compliant=float(np.sum(self._people.compliance))
            / len(self._people.compliance),
            percent_male=float(np.sum(self._people.sex_is_male))
            / len(self._people.compliance),
            L1=NumericArrayStat.from_array(self._people.blackfly.L1),
            L2=NumericArrayStat.from_array(self._people.blackfly.L2),
            L3=NumericArrayStat.from_array(self._people.blackfly.L3),
            ages=NumericArrayStat.from_array(self._people.ages),
            mf=NumericArrayStat.from_array(self._people.mf),
            male_worms=NumericArrayStat.from_array(self._people.worms.male),
            infertile_female_worms=NumericArrayStat.from_array(
                self._people.worms.infertile
            ),
            fertile_female_worms=NumericArrayStat.from_array(
                self._people.worms.fertile
            ),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )


def make_state_from_params(
    params: Params, n_people: int, gamma_distribution: float = 0.3
):
    return State.from_params(params, n_people, gamma_distribution=gamma_distribution)
