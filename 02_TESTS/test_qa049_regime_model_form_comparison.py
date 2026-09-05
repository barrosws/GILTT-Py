import math
import pytest

from gilttpy.analysis.regime_model_form import (
    StructuralAlternative, StructuralAxis, StructuralComparisonDesign,
    compare_structural_alternatives,
)
from gilttpy.validation.regime_model_form_campaign import (
    QA049_GATE, QA049_HOLDS, QA049_PROHIBITIONS,
    QA049_MATCHED_RESISTANCE_S_M, qa049_design, qa049_regime_definitions,
    qa049_boundary_law, run_qa049_campaign, qa049_max_mass_closure_abs,
    qa049_zero_settling_limit_difference, qa049_refinement_diagnostics,
)


def test_qa049_01_gate_holds_and_prohibitions_are_explicit():
    assert QA049_GATE == "PASS_TARGET_FREE_REGIME_AND_MODEL_FORM_COMPARISON"
    assert "HOLD_UNIVERSAL_AEROSOL_MODEL_FORM_DEFAULT" in QA049_HOLDS
    assert "HOLD_EMPIRICAL_MODEL_FORM_PROBABILITIES_AND_WEIGHTING" in QA049_HOLDS
    assert "PROHIBIT_TARGET_TUNING" in QA049_PROHIBITIONS
    assert "PROHIBIT_UNIVERSAL_ZHANG_VS_VP_WINNER" in QA049_PROHIBITIONS


def test_qa049_02_structural_design_rejects_targets_weights_and_winner_selection():
    r=(StructuralAlternative("r1","r1",StructuralAxis.REGIME,"p"),StructuralAlternative("r2","r2",StructuralAxis.REGIME,"p"))
    m=(StructuralAlternative("m1","m1",StructuralAxis.MODEL_FORM,"p"),StructuralAlternative("m2","m2",StructuralAxis.MODEL_FORM,"p"))
    for kwargs in (
        dict(observational_target_used=True),dict(probabilistic_weights_used=True),dict(winner_selection_requested=True)
    ):
        base=dict(label="d",provenance="p",qoi_units={"q":"1"},regimes=r,model_forms=m)
        base.update(kwargs)
        with pytest.raises(ValueError): StructuralComparisonDesign(**base)


def test_qa049_03_axis_separation_and_duplicate_keys_are_enforced():
    with pytest.raises(ValueError):
        StructuralComparisonDesign(
            "d","p",{"q":"1"},
            regimes=(StructuralAlternative("x","x",StructuralAxis.REGIME,"p"),StructuralAlternative("y","y",StructuralAxis.REGIME,"p")),
            model_forms=(StructuralAlternative("x","x",StructuralAxis.MODEL_FORM,"p"),StructuralAlternative("z","z",StructuralAxis.MODEL_FORM,"p")),
        )


def test_qa049_04_generic_structural_comparator_recovers_exact_discrete_contrasts():
    regimes=(StructuralAlternative("r1","r1",StructuralAxis.REGIME,"p"),StructuralAlternative("r2","r2",StructuralAxis.REGIME,"p"))
    models=(StructuralAlternative("m1","m1",StructuralAxis.MODEL_FORM,"p"),StructuralAlternative("m2","m2",StructuralAxis.MODEL_FORM,"p"))
    d=StructuralComparisonDesign("analytic","analytic",{"q":"1"},regimes,models)
    vals={('r1','m1'):2.,('r1','m2'):3.,('r2','m1'):4.,('r2','m2'):8.}
    out=compare_structural_alternatives(lambda r,m:{"q":vals[(r.key,m.key)]},d)
    assert out.contrast('r1','q').absolute_difference_b_minus_a == 1.0
    assert out.contrast('r1','q').symmetric_relative_difference_b_minus_a == pytest.approx(0.4)
    env=[x for x in out.regime_envelopes if x.model_form_key=='m1' and x.qoi_name=='q'][0]
    assert env.minimum==2 and env.maximum==4 and env.relative_span_to_max_abs==0.5


def test_qa049_05_regime_ladder_is_dimensionless_target_free_and_prespecified():
    regs=qa049_regime_definitions()
    assert [r.settling_resistance_number for r in regs] == [0.1,1.0,4.0]
    for r in regs:
        assert r.settling_velocity_m_s*QA049_MATCHED_RESISTANCE_S_M == pytest.approx(r.settling_resistance_number)
        assert "target-free" in r.provenance and "non-probabilistic" in r.provenance
    assert qa049_design().observational_target_used is False
    assert qa049_design().probabilistic_weights_used is False


def test_qa049_06_model_forms_share_the_exact_zero_settling_resistance_limit():
    assert qa049_zero_settling_limit_difference() < 1e-15


def test_qa049_07_boundary_formulas_are_distinct_only_after_nonzero_settling():
    for reg in qa049_regime_definitions():
        vp=qa049_boundary_law(reg.key,'vp1999'); zh=qa049_boundary_law(reg.key,'zhang2001_split')
        assert vp.settling_velocity_m_s == zh.settling_velocity_m_s == pytest.approx(reg.settling_velocity_m_s)
        assert vp.sink_velocity_m_s >= reg.settling_velocity_m_s
        assert zh.sink_velocity_m_s >= reg.settling_velocity_m_s
        assert vp.sink_velocity_m_s != pytest.approx(zh.sink_velocity_m_s,rel=1e-6)


def test_qa049_08_physical_campaign_evaluates_all_regime_form_pairs_and_closes_mass():
    r=run_qa049_campaign()
    assert len(r.evaluations)==6 and len(r.model_form_contrasts)==15
    assert qa049_max_mass_closure_abs(r) < 2e-9
    for row in r.evaluations:
        q=row.qois
        assert 0 < q['advective_survival_fraction'] < 1
        assert 0 < q['integrated_lower_loss_fraction'] < 1
        assert 10 < q['concentration_centroid_height_m'] < 110
        assert q['interface_total_exit_velocity_m_s'] > 0


def test_qa049_09_model_form_spread_is_regime_dependent_not_a_single_constant():
    r=run_qa049_campaign()
    vals=[abs(r.contrast(k,'integrated_lower_loss_fraction').symmetric_relative_difference_b_minus_a) for k in ('weak_settling','transition_settling','strong_settling')]
    assert min(vals) > 0.02
    assert max(vals)-min(vals) > 0.05
    # The order-one regime exposes a larger loss-fraction form contrast than either endpoint.
    assert vals[1] > vals[0] and vals[1] > vals[2]


def test_qa049_10_regime_changes_are_material_within_each_model_form():
    r=run_qa049_campaign()
    for model in ('vp1999','zhang2001_split'):
        weak=r.evaluation('weak_settling',model).qois['integrated_lower_loss_fraction']
        strong=r.evaluation('strong_settling',model).qois['integrated_lower_loss_fraction']
        assert strong > 5*weak


def test_qa049_11_refinement_preserves_qois_and_structural_contrasts():
    d=qa049_refinement_diagnostics()
    assert d['max_qoi_relative_change'] < 2e-6
    assert d['max_symmetric_contrast_absolute_change'] < 2e-6


def test_qa049_12_result_contract_exposes_no_probabilities_model_weights_or_winner():
    r=run_qa049_campaign()
    for forbidden in ('model_probabilities','model_weights','posterior_weights','best_model','winner','predictive_interval'):
        assert not hasattr(r,forbidden)
