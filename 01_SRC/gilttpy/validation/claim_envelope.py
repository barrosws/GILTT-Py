"""QA-045 validation evidence and release-claim envelope.

This validation-only module freezes what QA-040--QA-044 support, what remains a
controlled limitation, and what must not be claimed. It introduces no transport
physics and uses no observational target to define an acceptance claim.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    CONTROLLED_LIMITATION = "controlled_limitation"
    PROHIBITED = "prohibited"

@dataclass(frozen=True)
class ValidationClaim:
    key: str
    status: ClaimStatus
    statement: str
    evidence_checkpoints: tuple[str, ...]
    scope: str

CLAIMS: tuple[ValidationClaim, ...] = (
    ValidationClaim("constant_coeff_analytic", ClaimStatus.VERIFIED,
        "The conservative settling/deposition solver agrees with the independent corrected Ermak analytical reference in the frozen QA040 regimes.",
        ("QA040",), "constant u/K, tested finite-top and semi-infinite comparison regimes"),
    ValidationClaim("variable_coeff_exact_cross_discretization", ClaimStatus.VERIFIED,
        "The variable-coefficient conservative operator agrees with exact manufactured families and an independent P1 FEM reference in the frozen QA041-QA042 regimes.",
        ("QA041","QA042"), "tested manufactured coefficient families and gradient/interface stress campaign"),
    ValidationClaim("global_mass_conservation", ClaimStatus.VERIFIED,
        "The verified operator conserves represented source-to-outlet/surface mass under the QA039 budget identities.",
        ("QA039","QA040","QA041","QA042"), "represented resolved domain and explicitly represented aerosol diameter domain"),
    ValidationClaim("transient_causal_dehoog", ClaimStatus.VERIFIED,
        "The modern de Hoog path reproduces the exact causal variable-coefficient transient benchmark and independent direct-time cross-discretization within the frozen QA043 scope.",
        ("QA043",), "tested causal fronts, distances, times and complex128 model-evaluation contract"),
    ValidationClaim("inverse_laplace_policy", ClaimStatus.VERIFIED,
        "The modern inverse-Laplace policy is explicit de Hoog with degree 24 and working_dps 40; Fixed Talbot is historical/diagnostic only.",
        ("QA044",), "complex128 Laplace-space model evaluations; policy degree restricted to 8..30"),
    ValidationClaim("point_source_positivity", ClaimStatus.CONTROLLED_LIMITATION,
        "Finite-order global spectral point-source solutions can undershoot near the source; conservation and pointwise positivity are distinct.",
        ("QA040","QA041","QA042"), "near-source singular inlet representation"),
    ValidationClaim("temporal_discontinuity_accuracy", ClaimStatus.CONTROLLED_LIMITATION,
        "Accuracy immediately adjacent to a causal discontinuity is method/order sensitive and must be characterized separately.",
        ("QA043","QA044"), "temporal fronts and delayed transforms"),
    ValidationClaim("universal_positivity", ClaimStatus.PROHIBITED,
        "GILTT-Py 2.0 guarantees nonnegative pointwise concentration for every finite spectral truncation.",
        ("QA040","QA042"), "not supported"),
    ValidationClaim("universal_inverse_error_bound", ClaimStatus.PROHIBITED,
        "The frozen de Hoog policy has a universal error bound for every Laplace transform and time regime.",
        ("QA043","QA044"), "not supported"),
    ValidationClaim("empirical_field_validation_complete", ClaimStatus.PROHIBITED,
        "The modern GILTT-Py 2.0 deposition model has already been empirically validated against field observations.",
        ("QA040","QA041","QA042","QA043","QA044"), "QA040-QA044 are verification/benchmark checkpoints, not field validation"),
    ValidationClaim("universal_aerosol_scheme_winner", ClaimStatus.PROHIBITED,
        "Zhang-2001/Slinn or Venkatram-Pleim is universally the superior aerosol-deposition closure.",
        ("QA034","QA038","QA039"), "model-family selection remains application/model-form dependent"),
)

CHECKPOINT_TEST_COUNTS = {"QA040":190, "QA041":200, "QA042":210, "QA043":220, "QA044":230}


def claim(key: str) -> ValidationClaim:
    matches = [c for c in CLAIMS if c.key == key]
    if len(matches) != 1:
        raise KeyError(key)
    return matches[0]


def release_claims() -> tuple[ValidationClaim, ...]:
    return tuple(c for c in CLAIMS if c.status is ClaimStatus.VERIFIED)


def prohibited_claims() -> tuple[ValidationClaim, ...]:
    return tuple(c for c in CLAIMS if c.status is ClaimStatus.PROHIBITED)


def assert_claim_allowed(key: str) -> ValidationClaim:
    c = claim(key)
    if c.status is ClaimStatus.PROHIBITED:
        raise ValueError(f"claim is prohibited by QA045 evidence envelope: {key}")
    return c
