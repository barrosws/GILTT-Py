import math
import numpy as np

from gilttpy.analysis.global_sensitivity import variance_based_global_sensitivity, reestimate_prefix
from gilttpy.analysis.sensitivity import SensitivityAxis
from gilttpy.analysis.uncertainty import (
    DistributionFamily, ParametricUncertaintySpec, TargetFreeUncertaintyDesign,
    UncertaintyInterpretation, UncertaintyRepresentation,
)
from gilttpy.validation.global_sensitivity_campaign import (
    QA048_GATE, QA048_HOLDS, QA048_PROHIBITIONS, QA048_REFERENCE_N_BASE, QA048_LIVE_REPLAY_N_BASE, qa048_design, qa048_specs,
    run_qa048_campaign, qa048_index_ensemble_summary,
    qa048_dominant_total_order_by_scramble, qa048_mass_closure_max_abs,
)


def _spec(name):
    return ParametricUncertaintySpec(
        name, "1", 0.5, SensitivityAxis.PARAMETRIC,
        UncertaintyRepresentation.PROBABILITY_DISTRIBUTION,
        UncertaintyInterpretation.QA_REFERENCE,
        "analytic QA benchmark; no target",
        DistributionFamily.UNIFORM, (0.0, 1.0),
    )


def _design(names=("q",)):
    return TargetFreeUncertaintyDesign(
        "analytic", "analytic target-free benchmark", {n:"1" for n in names},
        True, "independent by analytic construction", False,
    )


def _orthogonal_model(p):
    z = {k: math.sqrt(12.0)*(p[k]-0.5) for k in ("x1","x2","x3")}
    y = (1.0*z["x1"] + 2.0*z["x2"] + 0.5*z["x3"]
         + 1.5*z["x1"]*z["x2"] + 0.75*z["x1"]*z["x3"]
         + 0.25*z["x2"]*z["x3"])
    return {"q": y}


def test_qa048_01_gate_holds_and_prohibitions_are_explicit():
    assert QA048_GATE == "PASS_TARGET_FREE_GLOBAL_SENSITIVITY_AND_INTERACTION_ANALYSIS"
    assert "HOLD_CORRELATED_INPUT_GLOBAL_SENSITIVITY" in QA048_HOLDS
    assert "HOLD_PHYSICAL_PAIRWISE_SECOND_ORDER_INTERACTION_QUANTIFICATION" in QA048_HOLDS
    assert "HOLD_REGIME_AND_MODEL_FORM_UNCERTAINTY_FOR_QA049" in QA048_HOLDS
    assert QA048_REFERENCE_N_BASE == 64 and QA048_LIVE_REPLAY_N_BASE == 32
    assert "PROHIBIT_ST_MINUS_S1_AS_PAIRWISE_INTERACTION" in QA048_PROHIBITIONS


def test_qa048_02_frozen_qa047_contract_is_reused_without_target_or_axis_mixing():
    specs=qa048_specs(); d=qa048_design()
    assert len(specs)==5 and d.observational_target_used is False and d.independent_inputs is True
    assert all(s.axis is SensitivityAxis.PARAMETRIC for s in specs)
    assert all(s.interpretation is UncertaintyInterpretation.QA_REFERENCE for s in specs)
    assert all(s.family is DistributionFamily.UNIFORM for s in specs)


def test_qa048_03_invalid_sample_size_and_interval_only_input_are_rejected():
    s=_spec("x")
    try:
        variance_based_global_sensitivity(lambda p:{"q":p["x"]}, (s,), _design(), n_base=12)
    except ValueError as e:
        assert "power of two" in str(e)
    else: raise AssertionError("non-power-of-two design accepted")
    interval=ParametricUncertaintySpec(
        "x","1",0.5,SensitivityAxis.PARAMETRIC,UncertaintyRepresentation.INTERVAL_ONLY,
        UncertaintyInterpretation.QA_REFERENCE,"interval only",None,(0.0,1.0))
    try:
        variance_based_global_sensitivity(lambda p:{"q":p["x"]}, (interval,), _design(), n_base=8)
    except ValueError as e:
        assert "probability distributions" in str(e)
    else: raise AssertionError("interval-only factor entered Sobol analysis")


def test_qa048_04_evaluator_contract_and_nonzero_output_variance_are_enforced():
    s=_spec("x")
    for evaluator in (lambda p:{"wrong":p["x"]}, lambda p:{"q":1.0}):
        try:
            variance_based_global_sensitivity(evaluator,(s,),_design(),n_base=8,seed=1)
        except ValueError: pass
        else: raise AssertionError("invalid evaluator was accepted")


def test_qa048_05_analytic_orthogonal_benchmark_recovers_first_total_and_pairwise_indices():
    specs=tuple(_spec(n) for n in ("x1","x2","x3"))
    r=variance_based_global_sensitivity(_orthogonal_model,specs,_design(),n_base=4096,seed=4801,workers=1)
    q=r.qoi("q"); V=8.125
    exact_s={"x1":1/V,"x2":4/V,"x3":0.25/V}
    exact_t={"x1":3.8125/V,"x2":6.3125/V,"x3":0.875/V}
    exact_s2={("x1","x2"):2.25/V,("x1","x3"):0.5625/V,("x2","x3"):0.0625/V}
    for name in exact_s:
        assert abs(q.factor(name).first_order-exact_s[name]) < 2e-3
        assert abs(q.factor(name).total_order-exact_t[name]) < 2e-3
    for pair, value in exact_s2.items():
        assert abs(q.pair(*pair).second_order-value) < 2e-3


def test_qa048_06_aggregate_interaction_involvement_is_st_minus_s1_but_pairwise_is_separate():
    specs=tuple(_spec(n) for n in ("x1","x2","x3"))
    r=variance_based_global_sensitivity(_orthogonal_model,specs,_design(),n_base=1024,seed=48)
    q=r.qoi("q"); f=q.factor("x1")
    assert abs(f.aggregate_interaction_involvement-(f.total_order-f.first_order)) < 1e-15
    assert q.pair("x1","x2").factor_a in ("x1","x2")
    assert not math.isclose(f.aggregate_interaction_involvement,q.pair("x1","x2").second_order,rel_tol=1e-2)


def test_qa048_07_prefix_reestimation_uses_no_new_calls_and_preserves_factor_contract():
    calls={"n":0}
    def f(p): calls["n"]+=1; return {"q":p["x"]+p["y"]+p["x"]*p["y"]}
    specs=(_spec("x"),_spec("y"))
    r=variance_based_global_sensitivity(f,specs,_design(),n_base=16,seed=7)
    before=calls["n"]; p=reestimate_prefix(r,8)
    assert calls["n"]==before and len(p)==1 and len(p[0].factors)==2
    assert r.n_model_evaluations==16*(2+2*2)


def test_qa048_08_physical_campaign_covers_four_qois_five_factors_and_first_total_design():
    r=run_qa048_campaign(QA048_LIVE_REPLAY_N_BASE,4801)
    assert r.n_base==32 and r.n_factors==5 and r.calc_second_order is False
    assert r.n_model_evaluations==224
    assert len(r.qoi_results)==4
    assert all(len(q.factors)==5 and len(q.second_order)==0 for q in r.qoi_results)


def test_qa048_09_every_physical_pick_freeze_evaluation_preserves_mass_budget():
    assert qa048_mass_closure_max_abs(run_qa048_campaign(QA048_LIVE_REPLAY_N_BASE,4801)) < 3e-9


def test_qa048_10_physical_indices_are_finite_and_total_effects_are_numerically_admissible():
    r=run_qa048_campaign(QA048_LIVE_REPLAY_N_BASE,4801)
    for q in r.qoi_results:
        assert q.output_variance>0 and math.isfinite(q.output_variance)
        for f in q.factors:
            assert math.isfinite(f.first_order) and math.isfinite(f.total_order)
            # finite-sample estimators may slightly cross theoretical [0,1] bounds.
            assert -0.25 < f.first_order < 1.25
            assert 0.0 <= f.total_order < 1.25


def test_qa048_11_randomized_qmc_scrambles_preserve_analytic_dominant_total_effect():
    specs=tuple(_spec(n) for n in ("x1","x2","x3"))
    for seed in (4811,4812,4813):
        r=variance_based_global_sensitivity(_orthogonal_model,specs,_design(),n_base=512,seed=seed)
        assert max(r.qoi("q").factors,key=lambda f:f.total_order).factor_name=="x2"


def test_qa048_12_campaign_contract_does_not_expose_model_form_weights_or_correlated_input_indices():
    r=run_qa048_campaign(QA048_LIVE_REPLAY_N_BASE,4801)
    assert not hasattr(r,"model_form_weights")
    assert not hasattr(r,"correlated_sobol_indices")
    assert r.calc_second_order is False  # physical pairwise Sij is not claimed by this gate
    assert qa048_design().independent_inputs is True
