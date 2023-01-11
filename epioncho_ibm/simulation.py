from typing import Iterator, overload

import numpy as np
import tqdm
from hdf5_dataclass import FileType

from epioncho_ibm.blackfly import (
    calc_l1,
    calc_l2,
    calc_l3,
    calc_new_worms_from_blackfly,
)
from epioncho_ibm.derived_params import DerivedParams
from epioncho_ibm.exposure import calculate_total_exposure
from epioncho_ibm.microfil import calculate_microfil_delta
from epioncho_ibm.params import Params
from epioncho_ibm.state import State
from epioncho_ibm.treatment import get_treatment
from epioncho_ibm.types import Array
from epioncho_ibm.worms import calculate_new_worms


class Simulation:
    state: State
    _derived_params: DerivedParams
    verbose: bool
    debug: bool

    @overload
    def __init__(
        self,
        *,
        start_time: float,
        params: Params,
        n_people: int,
        gamma_distribution: float = 0.3,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Create a new simulation, given the parameters.

        Args:
            start_time (float): Start time of the simulation
            params (Params): A set of fixed parameters for controlling the model.
            n_people (int): Number of people in the simulation
            gamma_distribution (float, optional): Individual level exposure heterogeneity. Defaults to 0.3.
            verbose (bool, optional): Verbose?. Defaults to False.
            debug (bool, optional): Debug?. Defaults to False.
        """
        ...

    @overload
    def __init__(
        self,
        *,
        input: FileType,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Restore the simulation from a previously saved file.

        Args:
            input (FileType): input file/stream
            verbose (bool, optional): Verbose?. Defaults to False.
            debug (bool, optional): Debug?. Defaults to False.
        """
        ...

    def __init__(
        self,
        *,
        params: Params | None = None,
        start_time: float | None = None,
        input: FileType | None = None,
        n_people: int | None = None,
        gamma_distribution: float | None = 0.3,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        assert (params is not None) != (
            input is not None
        ), "You must provide either `params` or `input`"
        if params:
            assert n_people is not None and gamma_distribution is not None
            self.state = State.from_params(
                params, n_people, gamma_distribution, start_time or 0.0
            )
        else:
            assert (
                input and start_time is None and not gamma_distribution and not n_people
            )
            # input
            self.state = State.from_hdf5(input)

        self._derive_params()
        self.verbose = verbose
        self.debug = debug

    def _derive_params(self) -> None:
        assert self.params
        self._derived_params = DerivedParams(self.params)

    @property
    def params(self) -> Params:
        return self.state.params

    @params.setter
    def params(self, params: Params):
        self.state.params = params
        self._derive_params()

    @property
    def derived_params(self) -> DerivedParams:
        assert self._derived_params
        return self._derived_params

    def reset_parameters(self, params: Params):
        """Reset the parameters

        Args:
            params (Params): New set of parameters
        """
        self.params = params

    def _advance(self):
        """Advance the state forward one time step from t to t + dt"""

        treatment = get_treatment(
            self.params.treatment,
            self.params.humans,
            self.params.delta_time,
            self.state.current_time,
            self._derived_params.treatment_times,
            self.state.people.ages,
            self.state.people.compliance,
        )

        total_exposure = calculate_total_exposure(
            self.params.exposure,
            self.state.people.ages,
            self.state.people.sex_is_male,
            self.state.people.individual_exposure,
        )
        self.state.people.ages += self.params.delta_time

        old_worms = self.state.people.worms.copy()

        # there is a delay in new parasites entering humans (from fly bites) and
        # entering the first adult worm age class
        new_worms = calc_new_worms_from_blackfly(
            self.state.people.blackfly.L3,
            self.params.blackfly,
            self.params.delta_time,
            total_exposure,
            self.state.n_people,
            old_worms,
            self.debug,
        )

        if self.state.people.delay_arrays.worm_delay is None:
            worm_delay: Array.Person.Int = new_worms
        else:
            worm_delay: Array.Person.Int = self.state.people.delay_arrays.worm_delay

        self.state.people.worms, last_time_of_last_treatment = calculate_new_worms(
            current_worms=self.state.people.worms,
            worm_params=self.params.worms,
            treatment=treatment,
            time_of_last_treatment=self.state.people.time_of_last_treatment,
            delta_time=self.params.delta_time,
            worm_delay_array=worm_delay,
            mortalities=self._derived_params.worm_mortality_rate,
            mortalities_generator=self._derived_params.worm_mortality_generator,
            current_time=self.state.current_time,
            debug=self.debug,
            worm_age_rate_generator=self._derived_params.worm_age_rate_generator,
            worm_sex_ratio_generator=self._derived_params.worm_sex_ratio_generator,
            worm_lambda_zero_generator=self._derived_params.worm_lambda_zero_generator,
            worm_omega_generator=self._derived_params.worm_omega_generator,
        )

        if (
            self.params.treatment is not None
            and self.state.current_time >= self.params.treatment.start_time
        ):
            self.state.people.time_of_last_treatment = last_time_of_last_treatment

        # inputs for delay in L1

        old_mf: Array.Person.Float = np.sum(self.state.people.mf, axis=0)
        self.state.people.mf += calculate_microfil_delta(
            current_microfil=self.state.people.mf,
            delta_time=self.params.delta_time,
            microfil_params=self.params.microfil,
            treatment_params=self.params.treatment,
            microfillarie_mortality_rate=self._derived_params.microfillarie_mortality_rate,
            fecundity_rates_worms=self._derived_params.fecundity_rates_worms,
            time_of_last_treatment=self.state.people.time_of_last_treatment,
            current_time=self.state.current_time,
            current_fertile_female_worms=old_worms.fertile,
            current_male_worms=old_worms.male,
            debug=self.debug,
        )
        old_blackfly_L1 = self.state.people.blackfly.L1

        if self.state.people.delay_arrays.exposure_delay is None:
            exposure_delay: Array.Person.Float = total_exposure
        else:
            exposure_delay: Array.Person.Float = (
                self.state.people.delay_arrays.exposure_delay
            )

        if self.state.people.delay_arrays.mf_delay is None:
            mf_delay: Array.Person.Float = old_mf
        else:
            mf_delay: Array.Person.Float = self.state.people.delay_arrays.mf_delay

        self.state.people.blackfly.L1 = calc_l1(
            self.params.blackfly,
            old_mf,
            mf_delay,
            total_exposure,
            exposure_delay,
            self.params.year_length_days,
        )

        old_blackfly_L2 = self.state.people.blackfly.L2
        self.state.people.blackfly.L2 = calc_l2(
            self.params.blackfly,
            old_blackfly_L1,
            mf_delay,
            exposure_delay,
            self.params.year_length_days,
        )
        self.state.people.blackfly.L3 = calc_l3(self.params.blackfly, old_blackfly_L2)
        # TODO: Resolve new_mf=old_mf
        self.state.people.delay_arrays.lag_all_arrays(
            new_worms=new_worms, total_exposure=total_exposure, new_mf=old_mf
        )
        people_to_die: Array.Person.Bool = np.logical_or(
            self._derived_params.people_to_die_generator.binomial(
                np.repeat(1, self.state.n_people)
            )
            == 1,
            self.state.people.ages >= self.params.humans.max_human_age,
        )
        self.state.people.process_deaths(people_to_die, self.params.humans.gender_ratio)

        self.state.current_time += self.params.delta_time

    def save(self, output: FileType) -> None:
        """Save the simulation to a file/stream.

        The output file will be in a HDF5 format. The simulation can then be
        restored with `Simulation.restore` class method.

        Args:
            output (FileType): output file/stream
        """
        self.state.to_hdf5(output)

    @classmethod
    def restore(cls, input: FileType) -> "Simulation":
        """Restore the simulation from a file/stream

        Args:
            input (FileType): HDF5 stream/file

        Returns:
            Simulation: restored simulation
        """
        return Simulation(input=input)

    @overload
    def iter_run(self, *, end_time: float, sampling_interval: float) -> Iterator[State]:
        """Run the simulation until `end_time`. Generates stats every `sampling_interval`,
        until `end_time`.

        This is a generator, so you must it as one.

        Examples:
            >>> simulation = Simulation(start_time=0, params=Params(), n_people=400)
            >>> [sample.mf_prevalence_in_population() for sample in simulation.iter_run(end_time=3, sampling_interval=1.0)]
            [0.99, 0.6, 0.2]

        Args:
            end_time (float): end time
            sampling_interval (float): State sampling interval (years)

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    @overload
    def iter_run(
        self, *, end_time: float, sampling_years: list[float]
    ) -> Iterator[State]:
        """Run the simulation until `end_time`. Generates stats for every year in `sampling_years`.

        This is a generator, so you must it as one.

        Examples:
            >>> simulation = Simulation(start_time=0, params=Params(), n_people=400)
            >>> for state in simulation.iter_run(end_time=10, sampling_years=[0.1, 1, 5])
            ...    print(state.mf_prevalence_in_population())
            0.99
            0.6
            0.2

        Args:
            end_time (float): end time
            sampling_years (list[float]): list of years to sample State

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    def iter_run(
        self,
        *,
        end_time: float,
        sampling_interval: float | None = None,
        sampling_years: list[float] | None = None,
    ) -> Iterator[State]:
        if end_time < self.state.current_time:
            raise ValueError("End time after start")

        if sampling_interval and sampling_years:
            raise ValueError(
                "You must provide sampling_interval, sampling_years or neither"
            )

        if sampling_years:
            sampling_years = sorted(sampling_years)

        sampling_years_idx = 0
        while self.state.current_time <= end_time:
            is_on_sampling_interval = (
                sampling_interval is not None
                and self.state.current_time % sampling_interval < self.params.delta_time
            )

            is_on_sampling_year = (
                sampling_years
                and sampling_years_idx < len(sampling_years)
                and abs(self.state.current_time - sampling_years[sampling_years_idx])
                < self.params.delta_time
            )

            if is_on_sampling_interval or is_on_sampling_year:
                yield self.state
                if is_on_sampling_year:
                    sampling_years_idx += 1

            self._advance()

    def run(self, *, end_time: float) -> None:
        """Run simulation from current state till `end_time`

        Args:
            end_time (float): end time of the simulation.
        """
        if end_time < self.state.current_time:
            raise ValueError("End time after start")

        # total progress bar must be a bit over so that the loop doesn't exceed total
        with tqdm.tqdm(
            total=end_time - self.state.current_time + self.params.delta_time,
            disable=not self.verbose,
        ) as progress_bar:
            while self.state.current_time <= end_time:
                progress_bar.update(self.params.delta_time)
                self._advance()
