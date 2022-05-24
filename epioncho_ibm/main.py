from . import Params
import numpy as np
from numpy.typing import NDArray

def generate_age_dist(
    population_size: int, 
    mean_human_age: float,
    max_age_person: float,
    delta_time: float, 
    limit: int = 75000
) -> NDArray[np.float64]:
    """
    Generate age distribution
    create inital age distribution and simulate stable age distribution
  
    """
    current_ages = np.zeros(population_size)
    delta_time_vector = np.ones(population_size)*delta_time
    for i in range(limit):
        current_ages += delta_time_vector
        death_vector = np.random.binomial(n = 1, p = (1/mean_human_age) * delta_time, size = population_size)
        np.place(current_ages, death_vector == 1 or current_ages >=max_age_person , 0)
    return current_ages

def weibull_mortality(
    delta_time: float,
    mu1: float,
    mu2: float,
    age_categories: np.ndarray
) -> np.ndarray:
    return delta_time * (mu1 ** mu2) * mu2 * (age_categories ** (mu2-1))

def ep_equilibrium_simulation(
    number_of_interations: int, #"time.its"
    annual_biting_rate: float, #"ABR"
    delta_time: float, #"DT"
    treatment_interval: float, #"treat.int"
    total_population_coverage: float, #"treat.prob"
    give_treatment: bool, #"give.treat"
    treatment_start_time: float, #"treat.start"
    treatment_stop_time: float, #"treat.stop"
    no_compliant_proportion: float, # "pnc" proportion of population which never receive treatment
    minimum_skin_snip_age: float, #"min.mont.age"
    params: Params,
    number_of_worm_age_classes: int
):
    worm_max_age = 20
    microfillarie_max_age = 2.5
    number_of_worm_age_categories = 20
    number_of_microfillariae_age_categories = 20

    current_ages = generate_age_dist(
        params.human_population,
        params.mean_human_age,
        params.max_age_person,
        delta_time
    )
    individual_exposure = np.random.gamma( #individual level exposure to fly bites
        shape = params.gamma_distribution, 
        scale = params.gamma_distribution, 
        size = params.human_population
    )
    if give_treatment:
        initial_treatment_times = np.arange( #"times.of.treat.in"
            start = treatment_start_time, 
            stop = (treatment_stop_time - treatment_interval/delta_time)
        )
    else:
        initial_treatment_times = 0

    #Cols to zero is a mechanism for resetting certain attributes to zero
    #columns_to_zero = np.arange( start = 1, stop = )

    #age-dependent mortality and fecundity rates of parasite life stages 
    worm_age_categories = np.arange(start = 0, stop = worm_max_age, step = worm_max_age) # age.cats
    worm_mortality_rate = weibull_mortality(delta_time, params.mu_worms1, params.mu_worms2, worm_age_categories)
    fecundity_rates_worms = 1.158305 * params.fecundity_worms_1 / (params.fecundity_worms_1 + (params.fecundity_worms_2 ** (-worm_age_categories)) - 1)

    microfillarie_age_categories = np.arange(start = 0, stop = microfillarie_max_age, step = microfillarie_max_age/number_of_microfillariae_age_categories) # age.cats.mf

    microfillarie_mortality_rate = weibull_mortality(delta_time, params.mu_microfillarie1, params.mu_microfillarie2, microfillarie_age_categories)


    #matrix for delay in L3 establishment in humans
    #RE-EVALUATE THIS SECTION
    #number_of_delay_cols = int(params.l3_delay * (28 / (delta_time*365)))
    #l_extras = np.zeros((number_of_delay_cols, params.human_population))
    #indices_l_mat = np.arange(2, number_of_delay_cols)

    # SET initial values in state

