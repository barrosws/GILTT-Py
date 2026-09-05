"""QA-048 target-free global sensitivity and interaction campaign.

The campaign inherits the frozen QA-047 QA/reference probability model exactly:
five independent PARAMETRIC inputs, Uniform marginals, no observational target,
and no numerical or model-form factors.  Therefore every Sobol' index is
conditional on that synthetic QA reference distribution and must not be presented
as universal physical importance.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from gilttpy.analysis.global_sensitivity import (
    GlobalSensitivityResult,
    reestimate_prefix,
    variance_based_global_sensitivity,
)
from gilttpy.validation.sensitivity_campaign import QA046ReferenceRegime, evaluate_qa046_qoi
from gilttpy.validation.uncertainty_campaign import qa047_design, qa047_specs


QA048_GATE = "PASS_TARGET_FREE_GLOBAL_SENSITIVITY_AND_INTERACTION_ANALYSIS"
QA048_HOLDS = (
    "HOLD_FIELD_DERIVED_INPUT_DISTRIBUTIONS",
    "HOLD_CORRELATED_INPUT_GLOBAL_SENSITIVITY",
    "HOLD_PHYSICAL_PAIRWISE_SECOND_ORDER_INTERACTION_QUANTIFICATION",
    "HOLD_REGIME_AND_MODEL_FORM_UNCERTAINTY_FOR_QA049",
    "HOLD_UNIVERSAL_PARAMETER_IMPORTANCE_CLAIM",
)
QA048_PROHIBITIONS = (
    "PROHIBIT_TARGET_TUNING",
    "PROHIBIT_NUMERICAL_OR_MODEL_FORM_FACTORS_IN_PARAMETRIC_SOBOL_SET",
    "PROHIBIT_QA_REFERENCE_SOBOL_INDICES_AS_UNIVERSAL_IMPORTANCE",
    "PROHIBIT_ST_MINUS_S1_AS_PAIRWISE_INTERACTION",
)
QA048_REFERENCE_N_BASE = 64
QA048_LIVE_REPLAY_N_BASE = 32
QA048_REFERENCE_SEED = 4801
QA048_REFERENCE_SEEDS = (4801, 4802, 4803)


def qa048_design():
    # Reuse the frozen QA047 probabilistic contract rather than redefining it.
    return qa047_design()


def qa048_specs():
    return qa047_specs()


def _evaluate(parameters):
    return evaluate_qa046_qoi(parameters, base=QA046ReferenceRegime())


@lru_cache(maxsize=4)
def run_qa048_campaign(
    n_base: int = QA048_REFERENCE_N_BASE,
    seed: int = QA048_REFERENCE_SEED,
    *,
    workers: int = 2,
) -> GlobalSensitivityResult:
    # Two Python workers are used only to reduce wall-clock time.  The evaluator
    # is deterministic and map order is preserved; no stochastic model state is introduced.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return variance_based_global_sensitivity(
            _evaluate, qa048_specs(), qa048_design(), n_base=n_base, seed=seed,
            calc_second_order=False, workers=1,
        )
    with threadpool_limits(limits=1):
        return variance_based_global_sensitivity(
            _evaluate, qa048_specs(), qa048_design(), n_base=n_base, seed=seed,
            calc_second_order=False, workers=workers,
        )


def qa048_prefix_results(result=None, n_prefix: int = 16):
    r = run_qa048_campaign() if result is None else result
    return reestimate_prefix(r, n_prefix)


def qa048_mass_closure_max_abs(result=None) -> float:
    r = run_qa048_campaign() if result is None else result
    residuals = []
    for store in (r.qoi_a, r.qoi_b):
        residuals.append(store["advective_survival_fraction"] + store["integrated_lower_loss_fraction"] - 1.0)
    residuals.append(r.qoi_ab["advective_survival_fraction"] + r.qoi_ab["integrated_lower_loss_fraction"] - 1.0)
    if r.calc_second_order:
        residuals.append(r.qoi_ba["advective_survival_fraction"] + r.qoi_ba["integrated_lower_loss_fraction"] - 1.0)
    return float(np.max(np.abs(np.concatenate([np.ravel(x) for x in residuals]))))


def run_qa048_scramble_ensemble(*, n_base: int = QA048_REFERENCE_N_BASE, workers: int = 2):
    """Three independent scrambled-QMC replicates for numerical-integration stability.

    The canonical QA048 freeze uses n_base=64.  n_base=32 is retained as a
    lighter live-replay diagnostic and as the exact nested prefix used in the
    32->64 convergence audit.
    """
    return tuple(run_qa048_campaign(n_base, seed, workers=workers) for seed in QA048_REFERENCE_SEEDS)


def qa048_index_ensemble_summary(results=None):
    """Mean and sample SD of S1, ST and raw ST-S1 across independent scrambles."""
    rs = run_qa048_scramble_ensemble() if results is None else tuple(results)
    if len(rs) < 2:
        raise ValueError("at least two independent scrambles are required")
    out = {}
    for qoi_name in qa048_design().qoi_units:
        out[qoi_name] = {}
        for factor in rs[0].factor_names:
            s1 = np.asarray([r.qoi(qoi_name).factor(factor).first_order for r in rs], dtype=float)
            st = np.asarray([r.qoi(qoi_name).factor(factor).total_order for r in rs], dtype=float)
            gap = st-s1
            out[qoi_name][factor] = {
                "first_order_mean": float(np.mean(s1)),
                "first_order_sd_across_scrambles": float(np.std(s1, ddof=1)),
                "total_order_mean": float(np.mean(st)),
                "total_order_sd_across_scrambles": float(np.std(st, ddof=1)),
                "interaction_involvement_mean": float(np.mean(gap)),
                "interaction_involvement_sd_across_scrambles": float(np.std(gap, ddof=1)),
            }
    return out


def qa048_dominant_total_order_by_scramble(results=None):
    rs = run_qa048_scramble_ensemble() if results is None else tuple(results)
    return {
        qoi: tuple(max(r.qoi(qoi).factors, key=lambda f: f.total_order).factor_name for r in rs)
        for qoi in qa048_design().qoi_units
    }
