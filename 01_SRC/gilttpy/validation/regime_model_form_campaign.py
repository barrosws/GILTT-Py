"""QA-049 target-free regime and aerosol model-form comparison campaign.

Regimes are prespecified using the dimensionless settling-resistance number
Lambda = Vg*R with R=50 s/m and Lambda in {0.1, 1, 4}.  These are structural QA
scenarios, not frequencies or probability bins.  Two already verified complete
resolved-interface aerosol families are compared with matched non-settling
resistance R:

- Venkatram-Pleim mass-consistent closure: k = Vg/[1-exp(-Vg R)]
- Zhang-2001/Slinn local split: k = Vg + 1/R

The forms have the same exact zero-settling limit 1/R.  Therefore their
nonzero-settling differences are structural rather than a resistance mismatch.
No observations or historical predictions enter any regime, model-form choice,
contrast, resolution or gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import numpy as np

from gilttpy.analysis.regime_model_form import (
    StructuralAlternative, StructuralAxis, StructuralComparisonDesign,
    compare_structural_alternatives,
)
from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.physics.aerosol_transport_coupling import (
    venkatram_pleim_resolved_interface_flux_law,
    zhang2001_split_resolved_interface_flux_law,
)
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.robustness_campaign import (
    GaussianInletProfile, advective_flux_from_coefficients,
    concentration_from_coefficients, integrated_state,
    lower_flux_from_coefficients, project_profile_to_system,
    propagated_coefficients,
)
from gilttpy.validation.sensitivity_campaign import QA046ReferenceRegime


QA049_GATE = "PASS_TARGET_FREE_REGIME_AND_MODEL_FORM_COMPARISON"
QA049_HOLDS = (
    "HOLD_FIELD_DERIVED_INPUT_DISTRIBUTIONS",
    "HOLD_CORRELATED_INPUT_GLOBAL_SENSITIVITY",
    "HOLD_PHYSICAL_PAIRWISE_SECOND_ORDER_INTERACTION_QUANTIFICATION",
    "HOLD_EMPIRICAL_MODEL_FORM_PROBABILITIES_AND_WEIGHTING",
    "HOLD_UNIVERSAL_AEROSOL_MODEL_FORM_DEFAULT",
)
QA049_PROHIBITIONS = (
    "PROHIBIT_TARGET_TUNING",
    "PROHIBIT_STRUCTURAL_ALTERNATIVES_AS_UNJUSTIFIED_RANDOM_VARIABLES",
    "PROHIBIT_MODEL_FORM_SPREAD_AS_EMPIRICAL_PREDICTIVE_INTERVAL",
    "PROHIBIT_UNIVERSAL_ZHANG_VS_VP_WINNER",
)

QA049_MATCHED_RESISTANCE_S_M = 50.0
QA049_ZHANG_RA_SUB_S_M = 20.0
QA049_ZHANG_RS_S_M = 30.0


@dataclass(frozen=True)
class QA049RegimeDefinition:
    key: str
    label: str
    settling_resistance_number: float
    provenance: str

    @property
    def settling_velocity_m_s(self) -> float:
        return float(self.settling_resistance_number/QA049_MATCHED_RESISTANCE_S_M)

    def __post_init__(self) -> None:
        if not str(self.key).strip() or not str(self.label).strip() or not str(self.provenance).strip():
            raise ValueError("QA049 regime metadata are required")
        lam = float(self.settling_resistance_number)
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError("settling_resistance_number must be finite and positive")


def qa049_regime_definitions() -> tuple[QA049RegimeDefinition, ...]:
    p = "QA049 dimensionless structural regime ladder; target-free and non-probabilistic"
    return (
        QA049RegimeDefinition("weak_settling", "weak settling-resistance regime", 0.1, p),
        QA049RegimeDefinition("transition_settling", "order-one settling-resistance regime", 1.0, p),
        QA049RegimeDefinition("strong_settling", "strong settling-resistance regime", 4.0, p),
    )


def qa049_regimes() -> tuple[StructuralAlternative, ...]:
    return tuple(StructuralAlternative(r.key, r.label, StructuralAxis.REGIME, r.provenance) for r in qa049_regime_definitions())


def qa049_model_forms() -> tuple[StructuralAlternative, ...]:
    p = "QA038-verified resolved-interface complete model-form family; no cross-scheme mixing"
    return (
        StructuralAlternative("vp1999", "Venkatram-Pleim mass-consistent resolved-interface closure", StructuralAxis.MODEL_FORM, p),
        StructuralAlternative("zhang2001_split", "Zhang-2001/Slinn local split resolved-settling closure", StructuralAxis.MODEL_FORM, p),
    )


_QOI_UNITS = {
    "interface_total_exit_velocity_m_s": "m s^-1",
    "advective_survival_fraction": "1",
    "integrated_lower_loss_fraction": "1",
    "local_lower_flux_per_m": "m^-1",
    "concentration_centroid_height_m": "m",
}


def qa049_design() -> StructuralComparisonDesign:
    return StructuralComparisonDesign(
        label="QA049 target-free regime and aerosol model-form comparison",
        provenance=(
            "Discrete structural QA alternatives only. No observation, historical prediction, likelihood, "
            "calibration residual, probability weight, or winner-selection objective is used."
        ),
        qoi_units=_QOI_UNITS,
        regimes=qa049_regimes(), model_forms=qa049_model_forms(),
        observational_target_used=False, probabilistic_weights_used=False,
        winner_selection_requested=False,
    )


def _definition(key: str) -> QA049RegimeDefinition:
    for x in qa049_regime_definitions():
        if x.key == key:
            return x
    raise KeyError(key)


def qa049_boundary_law(regime_key: str, model_form_key: str):
    reg = _definition(regime_key); vg = reg.settling_velocity_m_s
    if model_form_key == "vp1999":
        return venkatram_pleim_resolved_interface_flux_law(
            settling_velocity_m_s=vg,
            unresolved_resistance_s_m=QA049_MATCHED_RESISTANCE_S_M,
            provenance=f"QA049 {regime_key}: matched R=50 s/m; target-free",
        )
    if model_form_key == "zhang2001_split":
        return zhang2001_split_resolved_interface_flux_law(
            settling_velocity_m_s=vg,
            residual_aerodynamic_resistance_s_m=QA049_ZHANG_RA_SUB_S_M,
            surface_resistance_s_m=QA049_ZHANG_RS_S_M,
            provenance=f"QA049 {regime_key}: matched Ra_sub+Rs=50 s/m; target-free",
        )
    raise KeyError(model_form_key)


def _evaluate(regime: StructuralAlternative, model: StructuralAlternative, *, n_modes: int, n_quad: int) -> dict[str, float]:
    rd = _definition(regime.key); vg = rd.settling_velocity_m_s
    base = replace(QA046ReferenceRegime(), settling_velocity_m_s=vg, n_modes=int(n_modes), n_quad=int(n_quad))
    boundary = qa049_boundary_law(regime.key, model.key)
    system = assemble_settling_legendre_system(
        h=base.domain_top_m, n_modes=base.n_modes,
        wind=base.wind, diffusivity=base.diffusivity,
        source_height=base.source_height_m, emission_rate=1.0,
        settling_velocity_m_s=vg, boundary=boundary,
        n_quad=base.n_quad, z_lower=base.z_lower_m,
    )
    inlet = GaussianInletProfile(
        base.z_lower_m, base.domain_top_m, base.source_height_m, base.source_sigma_m,
        source_rate=1.0, label="QA049 smooth inlet",
        provenance="QA046/QA049 fixed source regularization; not a fitted bandwidth",
    )
    profile = inlet.profile(base.wind, n_quad=1024)
    y0 = project_profile_to_system(profile, system, n_quad=1024)
    yx = propagated_coefficients(system, y0, base.x_end_m)
    iy = integrated_state(system, y0, base.x_end_m)
    fin = advective_flux_from_coefficients(system, y0)
    fout = advective_flux_from_coefficients(system, yx)
    lower_integrated = lower_flux_from_coefficients(system, iy)
    lower_local = lower_flux_from_coefficients(system, yx)
    if fin <= 0.0:
        raise FloatingPointError("QA049 inlet advective flux must be positive")
    zq, wq = gauss_legendre_interval(base.z_lower_m, base.domain_top_m, 512)
    c = concentration_from_coefficients(system, yx, zq)
    c_int = float(np.sum(wq*c))
    if c_int <= 0.0:
        raise FloatingPointError("QA049 concentration integral must be positive")
    centroid = float(np.sum(wq*zq*c)/c_int)
    return {
        "interface_total_exit_velocity_m_s": float(boundary.sink_velocity_m_s),
        "advective_survival_fraction": float(fout/fin),
        "integrated_lower_loss_fraction": float(lower_integrated/fin),
        "local_lower_flux_per_m": float(lower_local/fin),
        "concentration_centroid_height_m": centroid,
    }


@lru_cache(maxsize=4)
def run_qa049_campaign(n_modes: int = 48, n_quad: int = 384):
    if int(n_modes) < 8 or int(n_quad) < 2*int(n_modes):
        raise ValueError("QA049 requires n_modes>=8 and n_quad>=2*n_modes")
    design = qa049_design()
    return compare_structural_alternatives(
        lambda regime, model: _evaluate(regime, model, n_modes=int(n_modes), n_quad=int(n_quad)),
        design,
    )


def qa049_max_mass_closure_abs(result=None) -> float:
    r = run_qa049_campaign() if result is None else result
    return max(abs(row.qois["advective_survival_fraction"] + row.qois["integrated_lower_loss_fraction"] - 1.0) for row in r.evaluations)


def qa049_zero_settling_limit_difference() -> float:
    vp = venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=0.0, unresolved_resistance_s_m=QA049_MATCHED_RESISTANCE_S_M,
        provenance="QA049 zero-settling identity",
    )
    zh = zhang2001_split_resolved_interface_flux_law(
        settling_velocity_m_s=0.0, residual_aerodynamic_resistance_s_m=QA049_ZHANG_RA_SUB_S_M,
        surface_resistance_s_m=QA049_ZHANG_RS_S_M, provenance="QA049 zero-settling identity",
    )
    return abs(vp.sink_velocity_m_s-zh.sink_velocity_m_s)


def qa049_refinement_diagnostics():
    coarse = run_qa049_campaign(48,384); fine = run_qa049_campaign(64,512)
    max_qoi_rel = 0.0
    for row in coarse.evaluations:
        f = fine.evaluation(row.regime_key,row.model_form_key)
        for name in _QOI_UNITS:
            a,b=row.qois[name],f.qois[name]
            max_qoi_rel=max(max_qoi_rel,abs(a-b)/max(abs(b),1e-15))
    max_contrast_abs = 0.0
    for c in coarse.model_form_contrasts:
        f=fine.contrast(c.regime_key,c.qoi_name)
        max_contrast_abs=max(max_contrast_abs,abs(c.symmetric_relative_difference_b_minus_a-f.symmetric_relative_difference_b_minus_a))
    return {"max_qoi_relative_change":max_qoi_rel,"max_symmetric_contrast_absolute_change":max_contrast_abs}
