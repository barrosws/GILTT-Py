from gilttpy.validation.claim_envelope import (
    CLAIMS, CHECKPOINT_TEST_COUNTS, ClaimStatus, assert_claim_allowed, claim,
    prohibited_claims, release_claims,
)
import pytest


def test_qa040_to_qa044_counts_are_monotone_and_frozen():
    assert CHECKPOINT_TEST_COUNTS == {"QA040":190,"QA041":200,"QA042":210,"QA043":220,"QA044":230}


def test_analytic_ermak_claim_is_scoped_verified():
    c=claim("constant_coeff_analytic"); assert c.status is ClaimStatus.VERIFIED and c.evidence_checkpoints == ("QA040",)


def test_variable_coefficient_claim_requires_exact_and_cross_discretization_evidence():
    c=claim("variable_coeff_exact_cross_discretization"); assert c.status is ClaimStatus.VERIFIED and c.evidence_checkpoints == ("QA041","QA042")


def test_transient_dehoog_claim_is_verified_but_scoped():
    c=claim("transient_causal_dehoog"); assert c.status is ClaimStatus.VERIFIED and "complex128" in c.scope


def test_inverse_policy_claim_preserves_fixed_talbot_as_nondefault():
    c=claim("inverse_laplace_policy"); assert c.status is ClaimStatus.VERIFIED and "degree 24" in c.statement and "Fixed Talbot" in c.statement


def test_point_source_positivity_is_limitation_not_failure_or_claim():
    c=claim("point_source_positivity"); assert c.status is ClaimStatus.CONTROLLED_LIMITATION


def test_universal_positivity_claim_is_prohibited():
    with pytest.raises(ValueError): assert_claim_allowed("universal_positivity")


def test_universal_inverse_error_bound_is_prohibited():
    with pytest.raises(ValueError): assert_claim_allowed("universal_inverse_error_bound")


def test_field_validation_complete_claim_is_prohibited():
    with pytest.raises(ValueError): assert_claim_allowed("empirical_field_validation_complete")


def test_release_envelope_contains_only_verified_and_no_target_tuning_shortcut():
    assert release_claims() and all(c.status is ClaimStatus.VERIFIED for c in release_claims())
    assert all(c.status is ClaimStatus.PROHIBITED for c in prohibited_claims())
    assert len({c.key for c in CLAIMS}) == len(CLAIMS)
