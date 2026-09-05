"""Stomatal-physiology providers for GILTT-Py 2.0.

QA-030D separates vegetation physiology from the QA-030C resistance-network combiner.
It implements traceable bulk-canopy response functions for light/PAR, temperature,
vapour-pressure deficit, soil water, phenology and LAI, together with explicit gas-
diffusivity scaling when a maximum conductance is defined for a reference gas.

Two contracts are intentionally distinguished:

1. DEPAC-like factor product at canopy scale::

       Gs = gmax_leaf * LAI * D_ratio * fphen * fsw * fvpd * fT * fPAR
       Rstom = 1/Gs

2. Jarvis-Emberson bulk multiplicative conductance::

       g_leaf = gmax_leaf * D_ratio * fphen * flight
                * max(fmin, fT * fVPD * fSW)
       G_canopy = LAI * g_leaf

This module is a big-leaf physiology provider, not a full multi-layer/sunlit-shaded
canopy model and not a land-use parameter database.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _fraction(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def ppfd_from_shortwave_w_m2(
    shortwave_down_w_m2: float,
    *,
    par_energy_fraction: float = 0.5,
    photon_conversion_umol_j: float = 4.5,
) -> float:
    """Convert shortwave irradiance to PPFD using an explicit CLRTAP-style convention.

    The common screening conversion is PPFD = SW * 0.5 * 4.5.  Both coefficients are
    exposed so they are never hidden calibration constants.
    """
    sw = _nonnegative("shortwave_down_w_m2", shortwave_down_w_m2)
    par_fraction = _fraction("par_energy_fraction", par_energy_fraction)
    conversion = _positive("photon_conversion_umol_j", photon_conversion_umol_j)
    return sw * par_fraction * conversion


def saturation_vapor_pressure_kpa_clrtap(temperature_c: float) -> float:
    """CLRTAP/ETC saturation-vapour-pressure approximation in kPa."""
    t = _finite("temperature_c", temperature_c)
    denominator = 240.97 + t
    if denominator <= 0.0:
        raise ValueError("temperature_c is outside the valid range of the saturation formula")
    return 0.61 * math.exp(17.502 * t / denominator)


def vapor_pressure_deficit_kpa(
    *, temperature_c: float, relative_humidity_fraction: float
) -> float:
    """Return VPD = es(T) * (1-RH), with RH supplied as a 0--1 fraction."""
    rh = _fraction("relative_humidity_fraction", relative_humidity_fraction)
    return saturation_vapor_pressure_kpa_clrtap(temperature_c) * (1.0 - rh)


def emberson_light_factor(
    ppfd_umol_m2_s: float, *, light_response_per_umol: float
) -> float:
    """Emberson/CLRTAP light factor: f_light = 1-exp(-a*PPFD)."""
    ppfd = _nonnegative("ppfd_umol_m2_s", ppfd_umol_m2_s)
    a = _nonnegative("light_response_per_umol", light_response_per_umol)
    if a == 0.0 or ppfd == 0.0:
        return 0.0
    return -math.expm1(-a * ppfd)


def cardinal_temperature_factor(
    temperature_c: float,
    *,
    minimum_c: float,
    optimum_c: float,
    maximum_c: float,
) -> float:
    """Cardinal-temperature stomatal response on [Tmin,Tmax].

    Uses the CLRTAP/DO3SE family
      ((T-Tmin)/(Topt-Tmin))*((Tmax-T)/(Tmax-Topt))**b,
    b=(Tmax-Topt)/(Topt-Tmin),
    and returns zero outside the open interval.  The response equals one at Topt.
    """
    t = _finite("temperature_c", temperature_c)
    tmin = _finite("minimum_c", minimum_c)
    topt = _finite("optimum_c", optimum_c)
    tmax = _finite("maximum_c", maximum_c)
    if not tmin < topt < tmax:
        raise ValueError("temperature cardinal points must satisfy minimum < optimum < maximum")
    if t <= tmin or t >= tmax:
        return 0.0
    exponent = (tmax - topt) / (topt - tmin)
    value = ((t - tmin) / (topt - tmin)) * ((tmax - t) / (tmax - topt)) ** exponent
    return min(1.0, max(0.0, value))


def linear_vpd_factor(
    vpd_kpa: float,
    *,
    full_open_kpa: float,
    minimum_open_kpa: float,
    minimum_fraction: float = 0.0,
) -> float:
    """Piecewise-linear VPD response using unambiguous endpoint names.

    f=1 at/below ``full_open_kpa`` and f=minimum_fraction at/above
    ``minimum_open_kpa``.  Literature often names these VPD_min and VPD_max,
    respectively; the API avoids that ambiguous semantic inversion.
    """
    vpd = _nonnegative("vpd_kpa", vpd_kpa)
    low = _nonnegative("full_open_kpa", full_open_kpa)
    high = _positive("minimum_open_kpa", minimum_open_kpa)
    floor = _fraction("minimum_fraction", minimum_fraction)
    if not low < high:
        raise ValueError("full_open_kpa must be smaller than minimum_open_kpa")
    if vpd <= low:
        return 1.0
    if vpd >= high:
        return floor
    x = (vpd - low) / (high - low)
    return 1.0 - x * (1.0 - floor)


def linear_soil_water_content_factor(
    soil_water_content_m3_m3: float,
    *,
    wilting_point_m3_m3: float,
    field_capacity_m3_m3: float,
    minimum_fraction: float = 0.0,
) -> float:
    """Linear soil-water stress factor between wilting point and field capacity."""
    theta = _nonnegative("soil_water_content_m3_m3", soil_water_content_m3_m3)
    wilt = _nonnegative("wilting_point_m3_m3", wilting_point_m3_m3)
    field = _positive("field_capacity_m3_m3", field_capacity_m3_m3)
    floor = _fraction("minimum_fraction", minimum_fraction)
    if not wilt < field:
        raise ValueError("wilting_point_m3_m3 must be smaller than field_capacity_m3_m3")
    if theta <= wilt:
        return floor
    if theta >= field:
        return 1.0
    raw = (theta - wilt) / (field - wilt)
    return max(floor, min(1.0, raw))


def fixed_day_phenology_factor(
    day_of_year: float,
    *,
    start_day: float,
    end_day: float,
    ramp_up_days: float = 0.0,
    ramp_down_days: float = 0.0,
    start_fraction: float = 1.0,
    end_fraction: float = 1.0,
    outside_fraction: float = 0.0,
) -> float:
    """Fixed-day growing-season phenology factor with linear edge ramps.

    This helper intentionally handles only a non-wrapping season with start_day <= end_day.
    Cross-year growing seasons require a separately audited calendar convention.
    """
    day = _finite("day_of_year", day_of_year)
    if not 1.0 <= day <= 366.0:
        raise ValueError("day_of_year must lie in [1, 366]")
    start = _finite("start_day", start_day)
    end = _finite("end_day", end_day)
    up = _nonnegative("ramp_up_days", ramp_up_days)
    down = _nonnegative("ramp_down_days", ramp_down_days)
    fstart = _fraction("start_fraction", start_fraction)
    fend = _fraction("end_fraction", end_fraction)
    fout = _fraction("outside_fraction", outside_fraction)
    if not 1.0 <= start <= end <= 366.0:
        raise ValueError("phenology dates must satisfy 1 <= start_day <= end_day <= 366")
    if up > end - start or down > end - start or up + down > end - start:
        raise ValueError("phenology ramps do not fit inside the growing season")
    if day < start or day > end:
        return fout
    if up > 0.0 and day < start + up:
        return fstart + (1.0 - fstart) * (day - start) / up
    if down > 0.0 and day > end - down:
        return fend + (1.0 - fend) * (end - day) / down
    return 1.0


def diffusivity_ratio(
    *, species_diffusivity_m2_s: float, reference_diffusivity_m2_s: float
) -> float:
    """Return D_species/D_reference for conductance scaling."""
    return _positive("species_diffusivity_m2_s", species_diffusivity_m2_s) / _positive(
        "reference_diffusivity_m2_s", reference_diffusivity_m2_s
    )


def depac_stomatal_resistance_from_factors(
    *,
    maximum_leaf_conductance_m_s: float,
    leaf_area_index: float,
    phenology_factor: float,
    soil_water_factor: float,
    vpd_factor: float,
    temperature_factor: float,
    par_factor: float,
    species_to_reference_diffusivity_ratio: float = 1.0,
) -> float:
    """DEPAC-like bulk stomatal resistance from already evaluated factors."""
    gmax = _nonnegative("maximum_leaf_conductance_m_s", maximum_leaf_conductance_m_s)
    lai = _nonnegative("leaf_area_index", leaf_area_index)
    dratio = _positive(
        "species_to_reference_diffusivity_ratio", species_to_reference_diffusivity_ratio
    )
    factors = [
        _fraction("phenology_factor", phenology_factor),
        _fraction("soil_water_factor", soil_water_factor),
        _fraction("vpd_factor", vpd_factor),
        _fraction("temperature_factor", temperature_factor),
        _fraction("par_factor", par_factor),
    ]
    conductance = gmax * lai * dratio * math.prod(factors)
    return math.inf if conductance == 0.0 else 1.0 / conductance


@dataclass(frozen=True)
class StomatalEnvironment:
    """Meteorological/soil state used by the bulk stomatal provider."""

    ppfd_umol_m2_s: float
    temperature_c: float
    vapor_pressure_deficit_kpa: float
    soil_water_content_m3_m3: float
    day_of_year: float
    leaf_area_index: float

    def __post_init__(self) -> None:
        _nonnegative("ppfd_umol_m2_s", self.ppfd_umol_m2_s)
        _finite("temperature_c", self.temperature_c)
        _nonnegative("vapor_pressure_deficit_kpa", self.vapor_pressure_deficit_kpa)
        _nonnegative("soil_water_content_m3_m3", self.soil_water_content_m3_m3)
        day = _finite("day_of_year", self.day_of_year)
        if not 1.0 <= day <= 366.0:
            raise ValueError("day_of_year must lie in [1, 366]")
        _nonnegative("leaf_area_index", self.leaf_area_index)


@dataclass(frozen=True)
class StomatalPhysiologyParameters:
    """Explicit vegetation parameter set; no hidden land-use lookup is performed."""

    vegetation_type: str
    maximum_leaf_conductance_m_s: float
    minimum_fraction: float
    light_response_per_umol: float
    temperature_minimum_c: float
    temperature_optimum_c: float
    temperature_maximum_c: float
    vpd_full_open_kpa: float
    vpd_minimum_open_kpa: float
    soil_wilting_point_m3_m3: float
    soil_field_capacity_m3_m3: float
    phenology_start_day: float = 1.0
    phenology_end_day: float = 366.0
    phenology_ramp_up_days: float = 0.0
    phenology_ramp_down_days: float = 0.0
    phenology_start_fraction: float = 1.0
    phenology_end_fraction: float = 1.0
    phenology_outside_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.vegetation_type).strip():
            raise ValueError("vegetation_type must be nonempty")
        _nonnegative("maximum_leaf_conductance_m_s", self.maximum_leaf_conductance_m_s)
        _fraction("minimum_fraction", self.minimum_fraction)
        _nonnegative("light_response_per_umol", self.light_response_per_umol)
        cardinal_temperature_factor(
            self.temperature_optimum_c,
            minimum_c=self.temperature_minimum_c,
            optimum_c=self.temperature_optimum_c,
            maximum_c=self.temperature_maximum_c,
        )
        linear_vpd_factor(
            self.vpd_full_open_kpa,
            full_open_kpa=self.vpd_full_open_kpa,
            minimum_open_kpa=self.vpd_minimum_open_kpa,
            minimum_fraction=self.minimum_fraction,
        )
        linear_soil_water_content_factor(
            self.soil_wilting_point_m3_m3,
            wilting_point_m3_m3=self.soil_wilting_point_m3_m3,
            field_capacity_m3_m3=self.soil_field_capacity_m3_m3,
            minimum_fraction=self.minimum_fraction,
        )
        fixed_day_phenology_factor(
            self.phenology_start_day,
            start_day=self.phenology_start_day,
            end_day=self.phenology_end_day,
            ramp_up_days=self.phenology_ramp_up_days,
            ramp_down_days=self.phenology_ramp_down_days,
            start_fraction=self.phenology_start_fraction,
            end_fraction=self.phenology_end_fraction,
            outside_fraction=self.phenology_outside_fraction,
        )


@dataclass(frozen=True)
class JarvisEmbersonBulkStomatalResistance:
    """Bulk big-leaf Jarvis-Emberson stomatal resistance provider.

    This provider evaluates explicit environmental response functions and applies the
    common multiplicative floor only to the T/VPD/soil-water product.  Light and
    phenology remain independent gates, so darkness or an explicitly zero phenology
    factor closes stomatal exchange even when ``minimum_fraction`` is positive.
    """

    environment: StomatalEnvironment
    parameters: StomatalPhysiologyParameters
    species_to_reference_diffusivity_ratio: float = 1.0

    def __post_init__(self) -> None:
        _positive(
            "species_to_reference_diffusivity_ratio", self.species_to_reference_diffusivity_ratio
        )

    @property
    def light_factor(self) -> float:
        return emberson_light_factor(
            self.environment.ppfd_umol_m2_s,
            light_response_per_umol=self.parameters.light_response_per_umol,
        )

    @property
    def temperature_factor(self) -> float:
        return cardinal_temperature_factor(
            self.environment.temperature_c,
            minimum_c=self.parameters.temperature_minimum_c,
            optimum_c=self.parameters.temperature_optimum_c,
            maximum_c=self.parameters.temperature_maximum_c,
        )

    @property
    def vpd_factor(self) -> float:
        return linear_vpd_factor(
            self.environment.vapor_pressure_deficit_kpa,
            full_open_kpa=self.parameters.vpd_full_open_kpa,
            minimum_open_kpa=self.parameters.vpd_minimum_open_kpa,
            minimum_fraction=self.parameters.minimum_fraction,
        )

    @property
    def soil_water_factor(self) -> float:
        return linear_soil_water_content_factor(
            self.environment.soil_water_content_m3_m3,
            wilting_point_m3_m3=self.parameters.soil_wilting_point_m3_m3,
            field_capacity_m3_m3=self.parameters.soil_field_capacity_m3_m3,
            minimum_fraction=self.parameters.minimum_fraction,
        )

    @property
    def phenology_factor(self) -> float:
        return fixed_day_phenology_factor(
            self.environment.day_of_year,
            start_day=self.parameters.phenology_start_day,
            end_day=self.parameters.phenology_end_day,
            ramp_up_days=self.parameters.phenology_ramp_up_days,
            ramp_down_days=self.parameters.phenology_ramp_down_days,
            start_fraction=self.parameters.phenology_start_fraction,
            end_fraction=self.parameters.phenology_end_fraction,
            outside_fraction=self.parameters.phenology_outside_fraction,
        )

    @property
    def environmental_floor_product(self) -> float:
        raw = self.temperature_factor * self.vpd_factor * self.soil_water_factor
        return max(self.parameters.minimum_fraction, raw)

    @property
    def leaf_conductance_m_s(self) -> float:
        return (
            self.parameters.maximum_leaf_conductance_m_s
            * self.species_to_reference_diffusivity_ratio
            * self.phenology_factor
            * self.light_factor
            * self.environmental_floor_product
        )

    @property
    def canopy_conductance_m_s(self) -> float:
        return self.environment.leaf_area_index * self.leaf_conductance_m_s

    def resistance_s_m(self) -> float:
        conductance = self.canopy_conductance_m_s
        return math.inf if conductance == 0.0 else 1.0 / conductance
