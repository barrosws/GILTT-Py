import math

from gilttpy.analysis.sensitivity import (
    DimensionedFactor,
    SensitivityAxis,
    TargetFreeSensitivityDesign,
    central_local_sensitivity,
    run_local_sensitivity_campaign,
)
from gilttpy.validation.sensitivity_campaign import (
    QA046_GATE,
    QA046_HOLDS,
    QA046ReferenceRegime,
    evaluate_qa046_qoi,
    qa046_baseline_parameters,
    qa046_design,
    qa046_factors,
    run_qa046_campaign,
)


def test_qa046_01_gate_and_future_scope_holds_are_explicit():
    assert QA046_GATE == "PASS_TARGET_FREE_DIMENSIONED_LOCAL_SENSITIVITY"
    assert QA046_HOLDS == (
        "HOLD_UNCERTAINTY_PROPAGATION_FOR_QA047",
        "HOLD_GLOBAL_INTERACTION_SENSITIVITY_FOR_QA048",
        "HOLD_REGIME_AND_MODEL_FORM_COMPARISON_FOR_QA049",
    )


def test_qa046_02_design_rejects_observational_target_use():
    try:
        TargetFreeSensitivityDesign("bad", "bad", {"q": "1"}, observational_target_used=True)
    except ValueError:
        pass
    else:
        raise AssertionError("QA046 must reject target-based sensitivity design")


def test_qa046_03_factor_requires_units_provenance_and_positive_parametric_baseline():
    for kwargs in (
        dict(name="", baseline=1.0, unit="m", axis=SensitivityAxis.PARAMETRIC, provenance="p"),
        dict(name="x", baseline=1.0, unit="", axis=SensitivityAxis.PARAMETRIC, provenance="p"),
        dict(name="x", baseline=1.0, unit="m", axis=SensitivityAxis.PARAMETRIC, provenance=""),
        dict(name="x", baseline=0.0, unit="m", axis=SensitivityAxis.PARAMETRIC, provenance="p"),
    ):
        try:
            DimensionedFactor(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dimensioned factor accepted")


def test_qa046_04_numerical_and_model_form_axes_cannot_be_pooled_as_local_derivatives():
    design = TargetFreeSensitivityDesign("d", "p", {"q": "1"})
    evaluator = lambda p: {"q": p["x"]}
    for axis in (SensitivityAxis.NUMERICAL, SensitivityAxis.MODEL_FORM):
        factor = DimensionedFactor("x", 2.0, "1", axis, "explicit separate axis")
        try:
            central_local_sensitivity(evaluator, {"x": 2.0}, factor, design)
        except ValueError:
            pass
        else:
            raise AssertionError("non-parametric axis was pooled into derivative sensitivity")


def test_qa046_05_central_elasticity_recovers_exact_power_law_to_second_order():
    design = TargetFreeSensitivityDesign("power", "analytic test", {"q": "kg"})
    factor = DimensionedFactor("x", 3.0, "m", SensitivityAxis.PARAMETRIC, "analytic", 0.01)
    est = central_local_sensitivity(lambda p: {"q": 7.0*p["x"]**2}, {"x": 3.0}, factor, design)[0]
    assert abs(est.elasticity-2.0) < 1e-12
    assert est.derivative_unit == "(kg)/(m)"


def test_qa046_06_campaign_rejects_mixed_uncertainty_axes():
    design = TargetFreeSensitivityDesign("d", "p", {"q": "1"})
    factors = (
        DimensionedFactor("x", 1.0, "m", SensitivityAxis.PARAMETRIC, "p"),
        DimensionedFactor("n", 10.0, "1", SensitivityAxis.NUMERICAL, "resolution"),
    )
    try:
        run_local_sensitivity_campaign(lambda p: {"q": p["x"]}, {"x": 1.0, "n": 10.0}, factors, design)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed uncertainty axes must remain separated")


def test_qa046_07_reference_regime_is_dimensioned_target_free_and_resolved():
    r = QA046ReferenceRegime()
    assert "no observational target" in r.provenance.lower()
    assert r.reference_wind_speed_m_s > 0.0
    assert r.diffusivity_lower_m2_s > 0.0
    assert r.settling_velocity_m_s > 0.0
    assert r.lower_sink_velocity_m_s > 0.0
    assert r.n_quad >= 2*r.n_modes
    assert qa046_design().observational_target_used is False


def test_qa046_08_baseline_transport_qoi_closes_mass_budget_without_target_fitting():
    q = evaluate_qa046_qoi(qa046_baseline_parameters())
    assert abs(q["advective_survival_fraction"] + q["integrated_lower_loss_fraction"] - 1.0) < 2e-9
    assert 0.0 < q["advective_survival_fraction"] < 1.0
    assert 0.0 < q["integrated_lower_loss_fraction"] < 1.0
    assert q["local_lower_flux_per_m"] > 0.0
    assert 10.0 < q["concentration_centroid_height_m"] < 110.0


def test_qa046_09_all_five_physical_factors_have_finite_dimensioned_derivatives_and_elasticities():
    c = run_qa046_campaign(perturbation_fraction=0.05)
    assert len(c.estimates) == 5*4
    assert {f.name for f in qa046_factors()} == {e.factor_name for e in c.estimates}
    assert all(math.isfinite(e.derivative) and math.isfinite(e.elasticity) for e in c.estimates)
    assert all(e.factor_unit and e.qoi_unit and e.derivative_unit for e in c.estimates)


def test_qa046_10_physical_directionality_is_consistent_for_integrated_lower_loss():
    c = run_qa046_campaign(perturbation_fraction=0.05)
    e = {x.factor_name: x.elasticity for x in c.for_qoi("integrated_lower_loss_fraction")}
    assert e["reference_wind_speed_m_s"] < 0.0
    assert e["settling_velocity_m_s"] > 0.0
    assert e["lower_sink_velocity_m_s"] > 0.0
    assert e["source_height_m"] < 0.0
    assert e["diffusivity_lower_m2_s"] > 0.0


def test_qa046_11_survival_and_integrated_loss_sensitivities_have_opposite_signs():
    c = run_qa046_campaign(perturbation_fraction=0.05)
    survival = {x.factor_name: x.elasticity for x in c.for_qoi("advective_survival_fraction")}
    loss = {x.factor_name: x.elasticity for x in c.for_qoi("integrated_lower_loss_fraction")}
    for name in survival:
        assert survival[name]*loss[name] < 0.0


def test_qa046_12_step_halving_confirms_local_derivative_stability_without_numerical_tuning():
    c5 = run_qa046_campaign(perturbation_fraction=0.05)
    c25 = run_qa046_campaign(perturbation_fraction=0.025)
    a = {(e.factor_name, e.qoi_name): e.elasticity for e in c5.estimates}
    b = {(e.factor_name, e.qoi_name): e.elasticity for e in c25.estimates}
    worst = max(abs(a[k]-b[k])/max(1.0, abs(b[k])) for k in a)
    assert worst < 0.005
