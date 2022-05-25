from pydantic import BaseModel

class Params(BaseModel):
    # ep.equi.sim parameters (bottom of 'R' file)
    timestep_count: int = 10  # total number of timesteps of the simulation
    bite_rate_per_person_per_year: float = (
        1000  # Annual biting rate 'ABR' in paper and in R code
    )
    treatment_intrvl_yrs: float = (
        1  # 'trt.int' treatment interval (years, 0.5 gives biannual)
    )
    timestep_size: float = 1 / 366  # the timestep ('DT.in' and 'DT' in code)
    treatment_probability: float = 0.65  # The probability that a 'treatable' person is actually treated in an iteration
    # unclear what gv.trt / give.treat is, given that it is '1'. Might be flag to enable or disable treatment logic
    treatment_start_iter: int = (
        0  # The iteration upon which treatment commences (treat.start in R code)
    )
    treatment_stop_iter: int = 0  # the iteration up which treatment stops (treat.stop)
    # 'pnc' or percentage non compliant is in random config
    min_skinsnip_age: int = 5  # TODO: below
    min_treatable_age: int = 5  # TODO: check if skinsnip and treatable age differ or whether they are always the same value

    # See line 476 R code

    # So-called Hard coded params
    # '.' have been replaced with '_'
    human_population: int = 440  # 'N' in R file

    # TODO: find out from client what the origin is of these values (delta.hz, delta.hinf, and c.h)
    delta_hz: float = 0.1864987  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → 0
    delta_hinf: float = 0.002772749  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → ∞
    c_h: float = 0.004900419  # Severity of transmission intensity dependent parasite establishment within humans

    human_blood_index: float = 0.63  # 'h' in paper, used in 'm' and 'beta' in R code
    recip_gono_cycle: float = 1 / 104  # 'g' in paper, used in 'm' and 'beta' in R code
    bite_rate_per_fly_on_human: float = (
        human_blood_index / recip_gono_cycle
    )  # defined in table D in paper, is 'beta' in R code
    annual_transm_potential: float = (
        bite_rate_per_person_per_year * human_population
    ) / bite_rate_per_fly_on_human  # ATP in doc, possible this is 'm' - implemented based on doc calculation
    blackfly_mort_per_person_per_year: float = (
        26  # Per capita mortality rate of blackfly vectors 'mu.v'
    )
    initial_mf: int = 0  # TODO: "int_mf" just 0, and doesn't change in program. What is the purpose?
    sigma_L0: int = 52  # TODO: unclear where this comes from, and what it means
    a_H: float = 0.8  # Time delay between L3 entering the host and establishing as adult worms in years
    # g is 'recip_gono_cycle'
    blackfly_mort_from_mf_per_person_per_year: float = (
        0.39  # Per capita microfilaria-induced mortality of blackfly vectors 'a.v'
    )
    max_human_age: int = 80  # 'real.max.age' in R file
    # human population is defined earlier
    mean_human_age: int = 50  # years 'mean.age' in R file

    # TODO: after establishing what int.L1/2/3 are, implement here
    lambda_zero: float = (
        1 / 3
    )  # Per capita rate of reversion from fertile to non-fertile adult female worms (lambda.zero / 0.33 in 'R' code)
    # omega
    omega: float = (
        0.59  # Per capita rate of progression from non-fertile to fertile adult female
    )
    delta_v_o: float = 0.0166  # TODO: verify with client, used in calc.L1
    c_v: float = 0.0205  # TODO: verify with client, used in calc.L1
    # sex.rat = 0.5 is in random config
    # num.mf.comps and num.comps.worm are both in the Person object

    # aging in parasites
    worms_aging: float = 1  # 'time.each.comp.worms'
    microfil_aging: float = 0.125  # 'time.each.comp.mf'
    microfil_move_rate: float = 8.13333  # 'mf.move.rate' #for aging in parasites 

    l1_l2_per_person_per_year: float = (
        201.6189  # Per capita development rate of larvae from stage L1 to L2 'nuone'
    )
    l2_l3_per_person_per_year: float = (
        207.7384  # Per capita development rate of larvae from stage L2 to L3 'nutwo'
    )
    initial_L3: float = 0.03 # "int.L3"
    initial_L2: float = 0.03 # "int.L2"
    initial_L1: float = 0.03 # "int.L1"
    initial_worms: int = 1 # "int.worms" initial number of worms in each worm age compartment

    skin_snip_weight: int = 2 # "ss.wt" the weight of the skin snip
    skin_snip_number: int = 2 # "num.ss"

    slope_kmf = 0.0478 # "slope.kmf"
    initial_kmf = 0.313 # "int.kMf"
    gamma_distribution = 0.3 #individual level exposure heterogeneity

    mu_worms1: float = 1.089 #parameters controlling age-dependent mortality in adult worms 
    mu_worms2: float = 6.00569 #parameters controlling age-dependent mortality in adult worms 
    mu_microfillarie1: float = 1.089 #parameters controlling age-dependent mortality in mf
    mu_microfillarie2: float = 1.428 #parameters controlling age-dependent mortality in mf
    fecundity_worms_1: float = 70
    fecundity_worms_2: float = 0.72 #parameters controlling age-dependent fecundity in adult worms 
    l3_delay: float = 10 # (days?) delay in worms entering humans and joining the first adult worm age class 

    delta_time: float = 1