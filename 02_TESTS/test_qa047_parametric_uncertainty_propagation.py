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
from gilttpy.validation.uncertainty_campaign import (
    QA047_GATE, QA047_HOLDS, qa047_design, qa047_specs,
    run_qa047_campaign, qa047_mass_closure_max_abs, qa047_delta_method_std,
)


def test_qa047_01_gate_and_holds_are_explicit():
    assert QA047_GATE == "PASS_TARGET_FREE_PARAMETRIC_UNCERTAINTY_PROPAGATION"
    assert QA047_HOLDS == (
        "HOLD_FIELD_DERIVED_INPUT_DISTRIBUTIONS",
        "HOLD_CORRELATED_INPUT_PROPAGATION",
        "HOLD_GLOBAL_VARIANCE_APPORTIONMENT_FOR_QA048",
        "HOLD_MODEL_FORM_AND_REGIME_UNCERTAINTY_FOR_QA049",
    )


def test_qa047_02_interval_only_knowledge_cannot_be_silently_uniformized():
    spec = ParametricUncertaintySpec(
        "x", "m", 2.0, SensitivityAxis.PARAMETRIC,
        UncertaintyRepresentation.INTERVAL_ONLY,
        UncertaintyInterpretation.EPISTEMIC_PROBABILISTIC,
        "interval from source; no PDF", None, (1.0, 3.0),
    )
    design = TargetFreeUncertaintyDesign("d", "p", {"q":"1"}, True, "independent test")
    try:
        propagate_parametric_uncertainty(lambda p:{"q":p["x"]}, (spec,), design, n_samples=8)
    except ValueError as e:
        assert "interval-only" in str(e)
    else:
        raise AssertionError("interval-only knowledge was silently probabilized")


def test_qa047_03_nonparametric_axes_are_rejected():
    try:
        ParametricUncertaintySpec(
            "n", "1", 32.0, SensitivityAxis.NUMERICAL,
            UncertaintyRepresentation.PROBABILITY_DISTRIBUTION,
            UncertaintyInterpretation.QA_REFERENCE,
            "numerical resolution", DistributionFamily.UNIFORM, (24.0, 40.0),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("numerical control entered parametric UQ")


def test_qa047_04_design_rejects_observational_target_and_implicit_correlation():
    for kwargs in (
        dict(label="d", provenance="p", qoi_units={"q":"1"}, independent_inputs=True,
             dependence_provenance="p", observational_target_used=True),
        dict(label="d", provenance="p", qoi_units={"q":"1"}, independent_inputs=False,
             dependence_provenance="correlation unknown", observational_target_used=False),
    ):
        try:
            TargetFreeUncertaintyDesign(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid QA047 design accepted")


def test_qa047_05_distribution_transforms_and_moments_are_exact_for_uniform_triangular_lognormal():
    A=SensitivityAxis.PARAMETRIC; U=UncertaintyRepresentation.PROBABILITY_DISTRIBUTION; Q=UncertaintyInterpretation.QA_REFERENCE
    specs = (
        ParametricUncertaintySpec("u","1",2,A,U,Q,"p",DistributionFamily.UNIFORM,(1,3)),
        ParametricUncertaintySpec("t","1",2,A,U,Q,"p",DistributionFamily.TRIANGULAR,(1,2,4)),
        ParametricUncertaintySpec("l","1",2,A,U,Q,"p",DistributionFamily.LOGNORMAL,(2,1.2)),
    )
    assert specs[0].analytic_mean_variance() == (2.0, 1.0/3.0)
    mt, vt = specs[1].analytic_mean_variance()
    assert abs(mt-7/3) < 1e-15 and vt > 0
    ml, vl = specs[2].analytic_mean_variance()
    assert ml > 2 and vl > 0
    for s in specs:
        x=s.transform_unit(np.array([0.1,0.5,0.9])); assert np.all(np.isfinite(x)) and np.all(x>0)


def test_qa047_06_randomized_sobol_propagation_recovers_linear_moments_without_targets():
    A=SensitivityAxis.PARAMETRIC; U=UncertaintyRepresentation.PROBABILITY_DISTRIBUTION; Q=UncertaintyInterpretation.QA_REFERENCE
    specs=(
        ParametricUncertaintySpec("x","m",2,A,U,Q,"analytic",DistributionFamily.UNIFORM,(1,3)),
        ParametricUncertaintySpec("y","s",4,A,U,Q,"analytic",DistributionFamily.UNIFORM,(2,6)),
    )
    d=TargetFreeUncertaintyDesign("linear","analytic",{"q":"kg"},True,"independent by construction")
    r=propagate_parametric_uncertainty(lambda p:{"q":2*p["x"]+3*p["y"]},specs,d,n_samples=1024,seed=47)
    exact_mean=2*2+3*4
    exact_var=4*(1/3)+9*(16/12)
    assert abs(r.summary("q").mean-exact_mean) < 2e-3
    assert abs(r.summary("q").standard_deviation-math.sqrt(exact_var)) < 2e-2


def test_qa047_07_reference_distributions_are_explicitly_qa_only_and_centered_on_qa046_baselines():
    specs=qa047_specs()
    assert len(specs)==5
    assert all(s.interpretation is UncertaintyInterpretation.QA_REFERENCE for s in specs)
    assert all("not a field-derived prior" in s.provenance for s in specs)
    for s in specs:
        mean,_=s.analytic_mean_variance()
        assert abs(mean-s.baseline) < 1e-14*max(1.0,abs(s.baseline))
    assert qa047_design().observational_target_used is False


def test_qa047_08_physical_campaign_propagates_all_four_qois_and_preserves_every_sample_mass_budget():
    r=run_qa047_campaign(64,4701)
    assert r.n_samples==64 and len(r.summaries)==4
    assert all(len(v)==64 for v in r.qoi_samples.values())
    assert qa047_mass_closure_max_abs(r) < 3e-9
    loss=r.qoi_samples["integrated_lower_loss_fraction"]
    assert np.all((loss>0)&(loss<1))
    cent=r.qoi_samples["concentration_centroid_height_m"]
    assert np.all((cent>10)&(cent<110))


def test_qa047_09_output_intervals_are_non_degenerate_and_physical():
    r=run_qa047_campaign(64,4701)
    for s in r.summaries:
        assert math.isfinite(s.mean) and s.standard_deviation>0
        assert s.minimum <= s.q025 <= s.median <= s.q975 <= s.maximum
    loss=r.summary("integrated_lower_loss_fraction")
    assert 0 < loss.q025 < loss.median < loss.q975 < 1


def test_qa047_10_sample_doubling_stabilizes_mean_and_standard_deviation_without_tuning():
    r32=run_qa047_campaign(32,4701); r64=run_qa047_campaign(64,4701)
    for q in qa047_design().qoi_units:
        a,b=r32.summary(q),r64.summary(q)
        mean_rel=abs(a.mean-b.mean)/max(abs(b.mean),1e-12)
        sd_rel=abs(a.standard_deviation-b.standard_deviation)/max(abs(b.standard_deviation),1e-12)
        assert mean_rel < 0.01
        assert sd_rel < 0.08


def test_qa047_11_direct_propagation_and_local_delta_method_agree_at_structural_uncertainty_scale():
    r=run_qa047_campaign(64,4701); delta=qa047_delta_method_std()
    # This is a nonlinearity diagnostic, not a variance-apportionment claim.
    for q, sd_lin in delta.items():
        sd_direct=r.summary(q).standard_deviation
        rel=abs(sd_direct-sd_lin)/sd_direct
        assert rel < 0.12


def test_qa047_12_no_global_sensitivity_or_model_form_ranking_is_exposed_by_campaign_contract():
    r=run_qa047_campaign(64,4701)
    assert not hasattr(r,"sobol_indices")
    assert not hasattr(r,"variance_contributions")
    assert not hasattr(r,"model_form_weights")
