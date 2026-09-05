"""QA-047 target-free parametric uncertainty propagation campaign.

The probability distributions in this module are QA/reference distributions used to
verify propagation and to characterize the nonlinear solver response. They are NOT
field-derived distributions for Copenhagen, Hanford, or any release validation site.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np

from gilttpy.analysis.sensitivity import SensitivityAxis
from gilttpy.analysis.uncertainty import (
    DistributionFamily,
    ParametricUncertaintySpec,
    TargetFreeUncertaintyDesign,
    UncertaintyInterpretation,
    UncertaintyRepresentation,
    propagate_parametric_uncertainty,
)
from gilttpy.validation.sensitivity_campaign import (
    QA046ReferenceRegime,
    evaluate_qa046_qoi,
    qa046_baseline_parameters,
    run_qa046_campaign,
)


QA047_GATE = "PASS_TARGET_FREE_PARAMETRIC_UNCERTAINTY_PROPAGATION"
QA047_HOLDS = (
    "HOLD_FIELD_DERIVED_INPUT_DISTRIBUTIONS",
    "HOLD_CORRELATED_INPUT_PROPAGATION",
    "HOLD_GLOBAL_VARIANCE_APPORTIONMENT_FOR_QA048",
    "HOLD_MODEL_FORM_AND_REGIME_UNCERTAINTY_FOR_QA049",
)

_QOI_UNITS = {
    "advective_survival_fraction": "1",
    "integrated_lower_loss_fraction": "1",
    "local_lower_flux_per_m": "m^-1",
    "concentration_centroid_height_m": "m",
}


def qa047_design() -> TargetFreeUncertaintyDesign:
    return TargetFreeUncertaintyDesign(
        label="QA047 target-free parametric uncertainty propagation",
        provenance=(
            "QA/reference probability distributions only; no observations, likelihood, residual, "
            "historical prediction, calibration score, or empirical performance target"
        ),
        qoi_units=_QOI_UNITS,
        independent_inputs=True,
        dependence_provenance=(
            "Independence is a deliberate QA/reference assumption for this synthetic campaign; "
            "it is not asserted for production atmospheric inputs"
        ),
        observational_target_used=False,
    )


def qa047_specs(regime: QA046ReferenceRegime | None = None) -> tuple[ParametricUncertaintySpec, ...]:
    r = QA046ReferenceRegime() if regime is None else regime
    p = "QA047 structural verification distribution; synthetic and not a field-derived prior"
    U = UncertaintyRepresentation.PROBABILITY_DISTRIBUTION
    Q = UncertaintyInterpretation.QA_REFERENCE
    A = SensitivityAxis.PARAMETRIC
    # Symmetric bounded ±10% QA distributions for transport scales; source height ±5 m.
    return (
        ParametricUncertaintySpec("reference_wind_speed_m_s", "m s^-1", r.reference_wind_speed_m_s, A, U, Q, p,
                                  DistributionFamily.UNIFORM, (0.90*r.reference_wind_speed_m_s, 1.10*r.reference_wind_speed_m_s)),
        ParametricUncertaintySpec("diffusivity_lower_m2_s", "m^2 s^-1", r.diffusivity_lower_m2_s, A, U, Q, p,
                                  DistributionFamily.UNIFORM, (0.90*r.diffusivity_lower_m2_s, 1.10*r.diffusivity_lower_m2_s)),
        ParametricUncertaintySpec("settling_velocity_m_s", "m s^-1", r.settling_velocity_m_s, A, U, Q, p,
                                  DistributionFamily.UNIFORM, (0.90*r.settling_velocity_m_s, 1.10*r.settling_velocity_m_s)),
        ParametricUncertaintySpec("lower_sink_velocity_m_s", "m s^-1", r.lower_sink_velocity_m_s, A, U, Q, p,
                                  DistributionFamily.UNIFORM, (0.90*r.lower_sink_velocity_m_s, 1.10*r.lower_sink_velocity_m_s)),
        ParametricUncertaintySpec("source_height_m", "m", r.source_height_m, A, U, Q, p,
                                  DistributionFamily.UNIFORM, (r.source_height_m-5.0, r.source_height_m+5.0)),
    )


@lru_cache(maxsize=4)
def run_qa047_campaign(n_samples: int = 64, seed: int = 4701):
    r = QA046ReferenceRegime()
    return propagate_parametric_uncertainty(
        lambda p: evaluate_qa046_qoi(p, base=r),
        qa047_specs(r), qa047_design(), n_samples=n_samples, seed=seed,
    )


def qa047_mass_closure_max_abs(result=None) -> float:
    out = run_qa047_campaign() if result is None else result
    s = out.qoi_samples["advective_survival_fraction"]
    l = out.qoi_samples["integrated_lower_loss_fraction"]
    return float(np.max(np.abs(s+l-1.0)))


def qa047_delta_method_std() -> dict[str, float]:
    """First-order local propagation diagnostic using QA046 derivatives and exact input variances."""
    local = run_qa046_campaign(perturbation_fraction=0.025)
    variances = {spec.name: spec.analytic_mean_variance()[1] for spec in qa047_specs()}
    out: dict[str, float] = {}
    for qoi in _QOI_UNITS:
        var = 0.0
        for est in local.for_qoi(qoi):
            var += est.derivative*est.derivative*variances[est.factor_name]
        out[qoi] = math.sqrt(var)
    return out
