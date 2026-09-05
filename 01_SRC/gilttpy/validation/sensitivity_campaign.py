"""QA-046 target-free local sensitivity campaign for the modern transport core.

The reference regime is synthetic and physically dimensioned.  It is not fit to
Copenhagen or any other observational target.  It uses a smooth Gaussian inlet
only to prevent the known point-source spectral undershoot from contaminating a
parameter-sensitivity gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import math
import numpy as np

from gilttpy.analysis.sensitivity import (
    DimensionedFactor,
    SensitivityAxis,
    TargetFreeSensitivityDesign,
    run_local_sensitivity_campaign,
)
from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.robustness_campaign import (
    GaussianInletProfile,
    advective_flux_from_coefficients,
    concentration_from_coefficients,
    integrated_state,
    lower_flux_from_coefficients,
    project_profile_to_system,
    propagated_coefficients,
)


QA046_GATE = "PASS_TARGET_FREE_DIMENSIONED_LOCAL_SENSITIVITY"
QA046_HOLDS = (
    "HOLD_UNCERTAINTY_PROPAGATION_FOR_QA047",
    "HOLD_GLOBAL_INTERACTION_SENSITIVITY_FOR_QA048",
    "HOLD_REGIME_AND_MODEL_FORM_COMPARISON_FOR_QA049",
)


@dataclass(frozen=True)
class QA046ReferenceRegime:
    z_lower_m: float = 10.0
    domain_top_m: float = 110.0
    reference_wind_speed_m_s: float = 2.4
    diffusivity_lower_m2_s: float = 1.5
    diffusivity_fractional_increase: float = 0.8
    settling_velocity_m_s: float = 0.054
    lower_sink_velocity_m_s: float = 0.024
    source_height_m: float = 60.0
    source_sigma_m: float = 5.0
    x_end_m: float = 1500.0
    n_modes: int = 48
    n_quad: int = 384
    provenance: str = (
        "QA046 synthetic dimensioned reference regime derived from previously verified "
        "variable-coefficient scales; no observational target or calibration objective"
    )

    def __post_init__(self) -> None:
        vals = (
            self.z_lower_m, self.domain_top_m, self.reference_wind_speed_m_s,
            self.diffusivity_lower_m2_s, self.diffusivity_fractional_increase,
            self.settling_velocity_m_s, self.lower_sink_velocity_m_s,
            self.source_height_m, self.source_sigma_m, self.x_end_m,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("QA046 regime parameters must be finite")
        if self.domain_top_m <= self.z_lower_m:
            raise ValueError("domain_top_m must exceed z_lower_m")
        if self.reference_wind_speed_m_s <= 0.0 or self.diffusivity_lower_m2_s <= 0.0:
            raise ValueError("wind and diffusivity scales must be positive")
        if self.diffusivity_fractional_increase <= 0.0:
            raise ValueError("diffusivity_fractional_increase must be positive")
        if self.settling_velocity_m_s < 0.0 or self.lower_sink_velocity_m_s < 0.0:
            raise ValueError("settling and lower sink velocities must be nonnegative")
        if not self.z_lower_m < self.source_height_m < self.domain_top_m:
            raise ValueError("source_height_m must lie strictly inside the domain")
        if self.source_sigma_m <= 0.0 or self.x_end_m <= 0.0:
            raise ValueError("source_sigma_m and x_end_m must be positive")
        if int(self.n_modes) < 8 or int(self.n_quad) < 2*int(self.n_modes):
            raise ValueError("insufficient QA046 spectral/quadrature resolution")
        if not str(self.provenance).strip():
            raise ValueError("provenance is required")

    @property
    def interval_length_m(self) -> float:
        return float(self.domain_top_m-self.z_lower_m)

    def wind(self, z):
        zz = np.asarray(z, dtype=np.float64)
        xi = (zz-self.z_lower_m)/self.interval_length_m
        return np.asarray(self.reference_wind_speed_m_s*(1.5-xi), dtype=np.float64)

    def diffusivity(self, z):
        zz = np.asarray(z, dtype=np.float64)
        xi = (zz-self.z_lower_m)/self.interval_length_m
        return np.asarray(
            self.diffusivity_lower_m2_s*(1.0+self.diffusivity_fractional_increase*xi),
            dtype=np.float64,
        )


_QOI_UNITS = {
    "advective_survival_fraction": "1",
    "integrated_lower_loss_fraction": "1",
    "local_lower_flux_per_m": "m^-1",
    "concentration_centroid_height_m": "m",
}


def qa046_design() -> TargetFreeSensitivityDesign:
    return TargetFreeSensitivityDesign(
        label="QA046 target-free dimensioned local transport sensitivity",
        provenance=(
            "No observations, likelihood, calibration residual, empirical target, or Copenhagen "
            "performance metric enters the design. Local central derivatives only."
        ),
        qoi_units=_QOI_UNITS,
        observational_target_used=False,
    )


def qa046_factors(regime: QA046ReferenceRegime | None = None) -> tuple[DimensionedFactor, ...]:
    r = QA046ReferenceRegime() if regime is None else regime
    p = "QA046 synthetic physical reference; target-free"
    return (
        DimensionedFactor("reference_wind_speed_m_s", r.reference_wind_speed_m_s, "m s^-1", SensitivityAxis.PARAMETRIC, p),
        DimensionedFactor("diffusivity_lower_m2_s", r.diffusivity_lower_m2_s, "m^2 s^-1", SensitivityAxis.PARAMETRIC, p),
        DimensionedFactor("settling_velocity_m_s", r.settling_velocity_m_s, "m s^-1", SensitivityAxis.PARAMETRIC, p),
        DimensionedFactor("lower_sink_velocity_m_s", r.lower_sink_velocity_m_s, "m s^-1", SensitivityAxis.PARAMETRIC, p),
        DimensionedFactor("source_height_m", r.source_height_m, "m", SensitivityAxis.PARAMETRIC, p),
    )


def qa046_baseline_parameters(regime: QA046ReferenceRegime | None = None) -> dict[str, float]:
    r = QA046ReferenceRegime() if regime is None else regime
    return {f.name: f.baseline for f in qa046_factors(r)}


def _regime_from_parameters(base: QA046ReferenceRegime, parameters: Mapping[str, float]) -> QA046ReferenceRegime:
    allowed = set(qa046_baseline_parameters(base))
    if set(parameters) != allowed:
        raise ValueError("QA046 evaluator parameter keys do not match frozen physical factors")
    return replace(base, **{k: float(v) for k, v in parameters.items()})


def evaluate_qa046_qoi(parameters: Mapping[str, float], *, base: QA046ReferenceRegime | None = None) -> dict[str, float]:
    """Evaluate four target-free transport quantities of interest."""
    frozen = QA046ReferenceRegime() if base is None else base
    r = _regime_from_parameters(frozen, parameters)
    system = assemble_settling_legendre_system(
        h=r.domain_top_m,
        n_modes=r.n_modes,
        wind=r.wind,
        diffusivity=r.diffusivity,
        source_height=r.source_height_m,
        emission_rate=1.0,
        settling_velocity_m_s=r.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(
            r.lower_sink_velocity_m_s,
            label="QA046 total lower flux; target-free physical sensitivity",
        ),
        n_quad=r.n_quad,
        z_lower=r.z_lower_m,
    )
    inlet = GaussianInletProfile(
        r.z_lower_m,
        r.domain_top_m,
        r.source_height_m,
        r.source_sigma_m,
        source_rate=1.0,
        label="QA046 smooth inlet",
        provenance="QA046 fixed source regularization; not a fitted bandwidth",
    )
    profile = inlet.profile(r.wind, n_quad=1024)
    y0 = project_profile_to_system(profile, system, n_quad=1024)
    yx = propagated_coefficients(system, y0, r.x_end_m)
    iy = integrated_state(system, y0, r.x_end_m)

    fin = advective_flux_from_coefficients(system, y0)
    fout = advective_flux_from_coefficients(system, yx)
    lower_integrated = lower_flux_from_coefficients(system, iy)
    lower_local = lower_flux_from_coefficients(system, yx)
    if fin <= 0.0:
        raise FloatingPointError("QA046 inlet advective flux must be positive")

    zq, wq = gauss_legendre_interval(r.z_lower_m, r.domain_top_m, 512)
    c = concentration_from_coefficients(system, yx, zq)
    c_int = float(np.sum(wq*c))
    if c_int <= 0.0:
        raise FloatingPointError("QA046 concentration integral must be positive")
    centroid = float(np.sum(wq*zq*c)/c_int)

    return {
        "advective_survival_fraction": float(fout/fin),
        "integrated_lower_loss_fraction": float(lower_integrated/fin),
        "local_lower_flux_per_m": float(lower_local/fin),
        "concentration_centroid_height_m": centroid,
    }


def run_qa046_campaign(*, perturbation_fraction: float = 0.05):
    regime = QA046ReferenceRegime()
    return run_local_sensitivity_campaign(
        lambda p: evaluate_qa046_qoi(p, base=regime),
        qa046_baseline_parameters(regime),
        qa046_factors(regime),
        qa046_design(),
        perturbation_fraction=perturbation_fraction,
    )
