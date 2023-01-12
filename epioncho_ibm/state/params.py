from typing import Optional

from pydantic import BaseModel


class BaseImmutableParams(BaseModel):
    class Config:
        allow_mutation = False


class TreatmentParams(BaseModel):
    # Treatment timing
    interval_years: float = 1  # treatment interval (years, 0.5 gives biannual)
    start_time: float = 100  # The iteration upon which treatment commences
    stop_time: float = 130  # the iteration upon which treatment stops

    # Affecting worms
    lam_m: float = 32.4  # effects of ivermectin
    phi: float = 19.6  # effects of ivermectin
    permanent_infertility: float = (
        0.345  # permenent infertility in worms due to ivermectin
    )

    # Affecting microfil
    u_ivermectin: float = 0.0096  # effects of ivermectin
    shape_parameter_ivermectin: float = 1.25  # effects of ivermectin

    # Affecting humans
    total_population_coverage: float = 0.65  # The probability that a 'treatable' person is actually treated in an iteration - "treat.prob"


class WormParams(BaseModel):
    mu_worms1: float = (
        0.09953  # parameters controlling age-dependent mortality in adult worms
    )
    mu_worms2: float = (
        6.00569  # parameters controlling age-dependent mortality in adult worms
    )
    initial_worms: int = 1  # initial number of worms in each worm age compartment
    worms_aging: float = 1  # the time worms spend in each age compartment
    worm_age_stages = 21
    max_worm_age = 21
    fecundity_worms_1: float = 70
    fecundity_worms_2: float = (
        0.72  # parameters controlling age-dependent fecundity in adult worms
    )
    omega: float = (
        0.59  # Per capita rate of progression from non-fertile to fertile adult female
    )
    lambda_zero: float = 0.33  # Per capita rate of reversion from fertile to non-fertile adult female worms
    sex_ratio: float = 0.5
    mf_production_per_worm: float = 1.158305  # Per capita rate of production of mf per mg of skin snip per fertile female adult worm at age 0


class BlackflyParams(BaseModel):
    delta_h_zero: float = 0.1864987  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → 0
    delta_h_inf: float = 0.002772749  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → ∞
    blackfly_mort_per_fly_per_year: float = (
        26  # Per capita mortality rate of blackfly vectors
    )
    blackfly_mort_from_mf_per_fly_per_year: float = (
        0.39  # Per capita microfilaria-induced mortality of blackfly vectors
    )
    mu_L3: int = 52  # Per capita mortality of L3 Larvae
    a_H: float = 0.8  # Proportion of infected larvae shed per bite a.H
    l1_l2_per_larva_per_year: float = (
        201.6189  # Per capita development rate of larvae from stage L1 to L2 'nuone'
    )
    l2_l3_per_larva_per_year: float = (
        207.7384  # Per capita development rate of larvae from stage L2 to L3 'nutwo'
    )
    delta_v0: float = 0.0166
    c_v: float = 0.0205

    initial_L3: float = 0.03  # "int.L3"
    initial_L2: float = 0.03  # "int.L2"
    initial_L1: float = 0.03  # "int.L1"

    human_blood_index: float = 0.63  # 'h' in paper, used in 'm' and 'beta' in R code
    gonotrophic_cycle_length: float = (
        1 / 104
    )  # 'g' in paper, used in 'm' and 'beta' in R code
    bite_rate_per_fly_on_human: float = human_blood_index / gonotrophic_cycle_length
    c_h: float = 0.004900419  # Severity of transmission intensity dependent parasite establishment within humans

    bite_rate_per_person_per_year: float = (
        1000  # Annual biting rate 'ABR' in paper and in R code
    )
    l1_delay: float = 4  # (days)
    l3_delay: float = 10  # "l3.delay" (months) delay in worms entering humans and joining the first adult worm age class
    immunity: float = 0.03962022  # strength of immunity
    with_immunity: bool = False


class MicrofilParams(BaseModel):
    microfil_move_rate: float = 8.13333  # 'mf.move.rate' #for aging in parasites
    microfil_age_stages = 21
    max_microfil_age = 2.5
    initial_mf: float = 0  # "int.mf"
    mu_microfillarie1: float = (
        1.089  # parameters controlling age-dependent mortality in mf
    )
    mu_microfillarie2: float = (
        1.428  # parameters controlling age-dependent mortality in mf
    )
    slope_kmf = 0.0478  # "slope.kmf"
    initial_kmf = 0.313  # "int.kMf"


class ExposureParams(BaseModel):
    # age-dependent exposure to fly bites
    male_exposure: float = 1.08  # "m.exp"
    female_exposure: float = 0.9  # "f.exp"
    male_exposure_exponent: float = 0.007  # "age.exp.m"
    female_exposure_exponent: float = -0.023  # "age.exp.f"


class HumanParams(BaseModel):
    max_human_age: int = 80  # 'real.max.age'
    mean_human_age: int = 50  # years 'mean.age'

    min_skinsnip_age: int = 5
    skin_snip_weight: int = 2  # "ss.wt" the weight of the skin snip
    skin_snip_number: int = 2  # "num.ss"
    gender_ratio: float = 0.5
    noncompliant_percentage: float = 0.05


class BaseParams(BaseModel):
    delta_time: float = 1 / 365  # DT
    year_length_days: float = 365
    month_length_days: float = 28


class Params(BaseParams):
    treatment: Optional[TreatmentParams] = TreatmentParams()
    worms: WormParams = WormParams()
    blackfly: BlackflyParams = BlackflyParams()
    microfil: MicrofilParams = MicrofilParams()
    exposure: ExposureParams = ExposureParams()
    humans: HumanParams = HumanParams()


class ImmutableTreatmentParams(TreatmentParams, BaseImmutableParams):
    pass


class ImmutableWormParams(WormParams, BaseImmutableParams):
    pass


class ImmutableBlackflyParams(BlackflyParams, BaseImmutableParams):
    pass


class ImmutableMicrofilParams(MicrofilParams, BaseImmutableParams):
    pass


class ImmutableExposureParams(ExposureParams, BaseImmutableParams):
    pass


class ImmutableHumanParams(HumanParams, BaseImmutableParams):
    pass


class ImmutableParams(BaseParams, BaseImmutableParams):
    treatment: Optional[ImmutableTreatmentParams] = ImmutableTreatmentParams()
    worms: ImmutableWormParams = ImmutableWormParams()
    blackfly: ImmutableBlackflyParams = ImmutableBlackflyParams()
    microfil: ImmutableMicrofilParams = ImmutableMicrofilParams()
    exposure: ImmutableExposureParams = ImmutableExposureParams()
    humans: ImmutableHumanParams = ImmutableHumanParams()


def immutable_to_mutable(immutable_params: ImmutableParams) -> Params:
    return Params.parse_obj(immutable_params.dict())


def mutable_to_immutable(params: Params) -> ImmutableParams:
    return ImmutableParams.parse_obj(params.dict())
