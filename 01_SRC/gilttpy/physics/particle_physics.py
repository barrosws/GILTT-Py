"""Aerosol particle-transport physics for GILTT-Py 2.0.

QA-030F supplies the particle state required by QA-030E without embedding a
surface-deposition model.  The module closes a traceable Stokes/Cunningham /
Brownian chain:

    lambda_air(T,p)  ->  Cc(dp,lambda)
    D_B = k_B T Cc / (3 pi mu dp)
    tau_p = rho_p dp**2 Cc / (18 mu)
    V_g = g tau_p
    Sc = nu_air / D_B

and the Zhang/Slinn Stokes-number conventions

    St_veg    = V_g u* / (g A) = tau_p u* / A
    St_smooth = V_g u*^2 / (g nu) = tau_p u*^2 / nu.

The factor g in the smooth-surface denominator is required dimensionally and
is present in later operational implementations; Zhang et al. (2001) is often
reproduced with a printed omission of this factor.  GILTT-Py does not implement
a dimensionful quantity under the name Stokes number.

The particle diameter is an explicit *current transport diameter*.  QA-030F
performs no hidden hygroscopic-growth calculation.  If a wet diameter is used,
its basis/provenance must be supplied by the caller and audited separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .deposition import G, cunningham_slip_correction
from .quasi_laminar import (
    air_dynamic_viscosity_sutherland_pa_s,
    air_kinematic_viscosity_sutherland_m2_s,
)

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
WILLEKE_REFERENCE_MEAN_FREE_PATH_M = 67.3e-9
WILLEKE_REFERENCE_TEMPERATURE_K = 296.15
WILLEKE_REFERENCE_PRESSURE_PA = 101_330.0
WILLEKE_SUTHERLAND_K = 110.4


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def air_mean_free_path_willeke_m(
    *,
    temperature_k: float,
    pressure_pa: float,
    reference_mean_free_path_m: float = WILLEKE_REFERENCE_MEAN_FREE_PATH_M,
    reference_temperature_k: float = WILLEKE_REFERENCE_TEMPERATURE_K,
    reference_pressure_pa: float = WILLEKE_REFERENCE_PRESSURE_PA,
    sutherland_k: float = WILLEKE_SUTHERLAND_K,
) -> float:
    """Mean free path of air using the Willeke/NIST T-p correction.

    lambda = lambda0*(T/T0)*(p0/p)*[(1+S/T0)/(1+S/T)]

    The default reference is 67.3 nm at 296.15 K and 101.33 kPa.  Reference
    values are exposed rather than hidden so alternate traceable conventions
    can be tested explicitly.
    """
    t = _positive("temperature_k", temperature_k)
    p = _positive("pressure_pa", pressure_pa)
    lam0 = _positive("reference_mean_free_path_m", reference_mean_free_path_m)
    t0 = _positive("reference_temperature_k", reference_temperature_k)
    p0 = _positive("reference_pressure_pa", reference_pressure_pa)
    s = _positive("sutherland_k", sutherland_k)
    return lam0 * (t / t0) * (p0 / p) * ((1.0 + s / t0) / (1.0 + s / t))


def brownian_diffusivity_m2_s(
    *,
    particle_diameter_m: float,
    temperature_k: float,
    air_dynamic_viscosity_pa_s: float,
    slip_correction: float,
    boltzmann_constant_j_k: float = BOLTZMANN_CONSTANT_J_K,
) -> float:
    """Stokes-Einstein-Cunningham particle diffusivity.

    D_B = k_B T Cc / (3*pi*mu*dp).
    """
    dp = _positive("particle_diameter_m", particle_diameter_m)
    t = _positive("temperature_k", temperature_k)
    mu = _positive("air_dynamic_viscosity_pa_s", air_dynamic_viscosity_pa_s)
    cc = _positive("slip_correction", slip_correction)
    kb = _positive("boltzmann_constant_j_k", boltzmann_constant_j_k)
    return kb * t * cc / (3.0 * math.pi * mu * dp)


def particle_relaxation_time_s(
    *,
    particle_diameter_m: float,
    particle_density_kg_m3: float,
    air_dynamic_viscosity_pa_s: float,
    slip_correction: float,
) -> float:
    """Stokes particle response time with Cunningham slip correction.

    tau_p = rho_p*dp**2*Cc/(18*mu).

    This is the same density convention used in the Zhang-2001 settling
    expression.  Air-buoyancy correction is not silently inserted.
    """
    dp = _positive("particle_diameter_m", particle_diameter_m)
    rho = _positive("particle_density_kg_m3", particle_density_kg_m3)
    mu = _positive("air_dynamic_viscosity_pa_s", air_dynamic_viscosity_pa_s)
    cc = _positive("slip_correction", slip_correction)
    return rho * dp * dp * cc / (18.0 * mu)


def stokes_settling_velocity_from_relaxation_m_s(
    *, particle_relaxation_time_s: float, gravity_m_s2: float = G
) -> float:
    """Return Zhang/Stokes gravitational settling speed ``Vg = g*tau_p``."""
    tau = _positive("particle_relaxation_time_s", particle_relaxation_time_s)
    g = _positive("gravity_m_s2", gravity_m_s2)
    return g * tau


def particle_schmidt_number(
    *, air_kinematic_viscosity_m2_s: float, brownian_diffusivity_m2_s: float
) -> float:
    """Particle Schmidt number ``Sc = nu_air / D_B``."""
    nu = _positive("air_kinematic_viscosity_m2_s", air_kinematic_viscosity_m2_s)
    db = _positive("brownian_diffusivity_m2_s", brownian_diffusivity_m2_s)
    return nu / db


def zhang2001_vegetated_stokes_number(
    *,
    settling_velocity_m_s: float,
    friction_velocity_m_s: float,
    collector_radius_m: float,
    gravity_m_s2: float = G,
) -> float:
    """Vegetated/rough-collector Stokes number ``St=Vg*u*/(g*A)``."""
    vg = _positive("settling_velocity_m_s", settling_velocity_m_s)
    ustar = _positive("friction_velocity_m_s", friction_velocity_m_s)
    radius = _positive("collector_radius_m", collector_radius_m)
    g = _positive("gravity_m_s2", gravity_m_s2)
    return vg * ustar / (g * radius)


def zhang2001_smooth_stokes_number_corrected(
    *,
    settling_velocity_m_s: float,
    friction_velocity_m_s: float,
    air_kinematic_viscosity_m2_s: float,
    gravity_m_s2: float = G,
) -> float:
    """Dimensionally consistent smooth-surface Stokes number.

    ``St = Vg*u*^2/(g*nu)``.

    The explicit ``g`` is retained because the frequently reproduced Zhang
    printed expression without it is not dimensionless.  Later operational
    documentation explicitly identifies the omission.
    """
    vg = _positive("settling_velocity_m_s", settling_velocity_m_s)
    ustar = _positive("friction_velocity_m_s", friction_velocity_m_s)
    nu = _positive("air_kinematic_viscosity_m2_s", air_kinematic_viscosity_m2_s)
    g = _positive("gravity_m_s2", gravity_m_s2)
    return vg * ustar * ustar / (g * nu)


@dataclass(frozen=True)
class AerosolParticleProperties:
    """Current particle state; diameter basis and provenance are mandatory."""

    diameter_m: float
    density_kg_m3: float
    diameter_basis: str
    provenance: str

    def __post_init__(self) -> None:
        _positive("diameter_m", self.diameter_m)
        _positive("density_kg_m3", self.density_kg_m3)
        if not str(self.diameter_basis).strip():
            raise ValueError("diameter_basis must be nonempty")
        if not str(self.provenance).strip():
            raise ValueError("provenance must be nonempty")


@dataclass(frozen=True)
class AerosolAirState:
    """Dry-air T-p state used to derive viscosity and mean free path."""

    temperature_k: float
    pressure_pa: float

    def __post_init__(self) -> None:
        _positive("temperature_k", self.temperature_k)
        _positive("pressure_pa", self.pressure_pa)

    @property
    def dynamic_viscosity_pa_s(self) -> float:
        return air_dynamic_viscosity_sutherland_pa_s(self.temperature_k)

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return air_kinematic_viscosity_sutherland_m2_s(
            temperature_k=self.temperature_k, pressure_pa=self.pressure_pa
        )

    @property
    def mean_free_path_m(self) -> float:
        return air_mean_free_path_willeke_m(
            temperature_k=self.temperature_k, pressure_pa=self.pressure_pa
        )


@dataclass(frozen=True)
class AerosolParticleTransportState:
    """Resolved particle-transport quantities required by collection schemes."""

    particle: AerosolParticleProperties
    air: AerosolAirState

    @property
    def slip_correction(self) -> float:
        return cunningham_slip_correction(
            self.particle.diameter_m, self.air.mean_free_path_m
        )

    @property
    def brownian_diffusivity_m2_s(self) -> float:
        return brownian_diffusivity_m2_s(
            particle_diameter_m=self.particle.diameter_m,
            temperature_k=self.air.temperature_k,
            air_dynamic_viscosity_pa_s=self.air.dynamic_viscosity_pa_s,
            slip_correction=self.slip_correction,
        )

    @property
    def relaxation_time_s(self) -> float:
        return particle_relaxation_time_s(
            particle_diameter_m=self.particle.diameter_m,
            particle_density_kg_m3=self.particle.density_kg_m3,
            air_dynamic_viscosity_pa_s=self.air.dynamic_viscosity_pa_s,
            slip_correction=self.slip_correction,
        )

    @property
    def settling_velocity_m_s(self) -> float:
        return stokes_settling_velocity_from_relaxation_m_s(
            particle_relaxation_time_s=self.relaxation_time_s
        )

    @property
    def schmidt_number(self) -> float:
        return particle_schmidt_number(
            air_kinematic_viscosity_m2_s=self.air.kinematic_viscosity_m2_s,
            brownian_diffusivity_m2_s=self.brownian_diffusivity_m2_s,
        )

    def stokes_number(
        self,
        *,
        friction_velocity_m_s: float,
        surface_regime: Literal["vegetated", "smooth"],
        collector_radius_m: float | None = None,
    ) -> float:
        if surface_regime == "vegetated":
            if collector_radius_m is None:
                raise ValueError("collector_radius_m is required for vegetated Stokes number")
            return zhang2001_vegetated_stokes_number(
                settling_velocity_m_s=self.settling_velocity_m_s,
                friction_velocity_m_s=friction_velocity_m_s,
                collector_radius_m=collector_radius_m,
            )
        if surface_regime == "smooth":
            if collector_radius_m is not None:
                raise ValueError("collector_radius_m is not used for smooth Stokes number")
            return zhang2001_smooth_stokes_number_corrected(
                settling_velocity_m_s=self.settling_velocity_m_s,
                friction_velocity_m_s=friction_velocity_m_s,
                air_kinematic_viscosity_m2_s=self.air.kinematic_viscosity_m2_s,
            )
        raise ValueError("surface_regime must be 'vegetated' or 'smooth'")
