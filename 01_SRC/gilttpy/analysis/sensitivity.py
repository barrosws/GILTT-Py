"""Target-free local sensitivity infrastructure for GILTT-Py 2.0.

QA-046 deliberately implements *local* sensitivity, not global sensitivity and
not uncertainty propagation.  For a positive physical factor ``theta`` and a
scalar quantity of interest ``Q``, the dimensioned central derivative is

    dQ/dtheta ~= [Q(theta(1+d))-Q(theta(1-d))] / (2 d theta),

and the corresponding local elasticity is

    E_theta^Q = (theta/Q) dQ/dtheta.

The elasticity is dimensionless and therefore permits ranking across physical
parameters that carry different units.  The dimensioned derivative is retained
alongside it so the physical scale is never discarded.

Governance is explicit:
- PARAMETRIC factors are physically dimensioned model inputs and may enter the
  local derivative screen;
- NUMERICAL factors (resolution, quadrature, inverse-Laplace degree, etc.) are
  convergence controls and are not pooled with physical sensitivities;
- MODEL_FORM factors are categorical alternatives and cannot be differentiated
  as though they were continuous parameters.

No observational target, calibration score, likelihood or fitted residual is
accepted by this module.  QA-047 handles uncertainty propagation; QA-048 may
add global/interacting sensitivity; QA-049 handles regime/model-form contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence
import math


class SensitivityAxis(str, Enum):
    """Mutually exclusive uncertainty/sensitivity axes."""

    PARAMETRIC = "parametric"
    NUMERICAL = "numerical"
    MODEL_FORM = "model_form"


def _nonempty(name: str, value: str) -> str:
    out = str(value).strip()
    if not out:
        raise ValueError(f"{name} is required")
    return out


def _finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class DimensionedFactor:
    """One auditable input factor with physical units and provenance."""

    name: str
    baseline: float
    unit: str
    axis: SensitivityAxis
    provenance: str
    perturbation_fraction: float = 0.05

    def __post_init__(self) -> None:
        _nonempty("name", self.name)
        _nonempty("unit", self.unit)
        _nonempty("provenance", self.provenance)
        baseline = _finite("baseline", self.baseline)
        if not isinstance(self.axis, SensitivityAxis):
            raise TypeError("axis must be a SensitivityAxis")
        d = _finite("perturbation_fraction", self.perturbation_fraction)
        if not 0.0 < d < 1.0:
            raise ValueError("perturbation_fraction must lie in (0,1)")
        if self.axis is SensitivityAxis.PARAMETRIC and baseline <= 0.0:
            raise ValueError("multiplicative parametric factors must have positive baseline")


@dataclass(frozen=True)
class TargetFreeSensitivityDesign:
    """Frozen design declaration for a local target-free sensitivity campaign."""

    label: str
    provenance: str
    qoi_units: Mapping[str, str]
    observational_target_used: bool = False

    def __post_init__(self) -> None:
        _nonempty("label", self.label)
        _nonempty("provenance", self.provenance)
        if self.observational_target_used:
            raise ValueError("QA-046 forbids observational targets or calibration objectives")
        if not self.qoi_units:
            raise ValueError("at least one quantity of interest is required")
        for name, unit in self.qoi_units.items():
            _nonempty("qoi name", name)
            _nonempty("qoi unit", unit)


@dataclass(frozen=True)
class LocalSensitivityEstimate:
    factor_name: str
    factor_unit: str
    qoi_name: str
    qoi_unit: str
    baseline_factor: float
    baseline_qoi: float
    perturbation_fraction: float
    qoi_minus: float
    qoi_plus: float
    derivative: float
    elasticity: float

    @property
    def derivative_unit(self) -> str:
        return f"({self.qoi_unit})/({self.factor_unit})"


@dataclass(frozen=True)
class SensitivityCampaign:
    design: TargetFreeSensitivityDesign
    baseline_parameters: Mapping[str, float]
    baseline_qoi: Mapping[str, float]
    estimates: tuple[LocalSensitivityEstimate, ...]

    def for_qoi(self, qoi_name: str) -> tuple[LocalSensitivityEstimate, ...]:
        return tuple(e for e in self.estimates if e.qoi_name == qoi_name)

    def for_factor(self, factor_name: str) -> tuple[LocalSensitivityEstimate, ...]:
        return tuple(e for e in self.estimates if e.factor_name == factor_name)

    def ranked(self, qoi_name: str) -> tuple[LocalSensitivityEstimate, ...]:
        return tuple(sorted(self.for_qoi(qoi_name), key=lambda e: abs(e.elasticity), reverse=True))


Evaluator = Callable[[Mapping[str, float]], Mapping[str, float]]


def _validate_qoi(result: Mapping[str, float], qoi_units: Mapping[str, str]) -> dict[str, float]:
    if set(result) != set(qoi_units):
        raise ValueError("evaluator QOI keys must exactly match design qoi_units")
    out: dict[str, float] = {}
    for name in qoi_units:
        out[name] = _finite(f"QOI {name}", result[name])
    return out


def central_local_sensitivity(
    evaluator: Evaluator,
    baseline_parameters: Mapping[str, float],
    factor: DimensionedFactor,
    design: TargetFreeSensitivityDesign,
    *,
    perturbation_fraction: float | None = None,
) -> tuple[LocalSensitivityEstimate, ...]:
    """Estimate dimensioned derivatives and elasticities for one physical factor.

    Only ``PARAMETRIC`` factors are admissible.  Numerical controls and model
    form alternatives must be handled by their own QA gates rather than pooled
    into a pseudo-parameter ranking.
    """
    if factor.axis is not SensitivityAxis.PARAMETRIC:
        raise ValueError("local derivative sensitivity is restricted to PARAMETRIC factors")
    if factor.name not in baseline_parameters:
        raise KeyError(f"factor {factor.name!r} missing from baseline_parameters")
    theta0 = _finite("factor baseline", baseline_parameters[factor.name])
    if theta0 <= 0.0:
        raise ValueError("multiplicative local sensitivity requires positive baseline")
    if not math.isclose(theta0, factor.baseline, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("factor baseline must exactly match baseline_parameters")
    d = factor.perturbation_fraction if perturbation_fraction is None else _finite(
        "perturbation_fraction", perturbation_fraction
    )
    if not 0.0 < d < 1.0:
        raise ValueError("perturbation_fraction must lie in (0,1)")

    base = {k: _finite(f"parameter {k}", v) for k, v in baseline_parameters.items()}
    minus = dict(base); plus = dict(base)
    minus[factor.name] = theta0*(1.0-d)
    plus[factor.name] = theta0*(1.0+d)

    q0 = _validate_qoi(evaluator(base), design.qoi_units)
    qm = _validate_qoi(evaluator(minus), design.qoi_units)
    qp = _validate_qoi(evaluator(plus), design.qoi_units)

    estimates: list[LocalSensitivityEstimate] = []
    denominator_theta = 2.0*d*theta0
    for name, unit in design.qoi_units.items():
        derivative = (qp[name]-qm[name])/denominator_theta
        if q0[name] == 0.0:
            raise ZeroDivisionError(
                f"QOI {name!r} has zero baseline; elasticity requires an explicit nonzero scale"
            )
        elasticity = derivative*theta0/q0[name]
        estimates.append(LocalSensitivityEstimate(
            factor_name=factor.name,
            factor_unit=factor.unit,
            qoi_name=name,
            qoi_unit=unit,
            baseline_factor=theta0,
            baseline_qoi=q0[name],
            perturbation_fraction=d,
            qoi_minus=qm[name],
            qoi_plus=qp[name],
            derivative=float(derivative),
            elasticity=float(elasticity),
        ))
    return tuple(estimates)


def run_local_sensitivity_campaign(
    evaluator: Evaluator,
    baseline_parameters: Mapping[str, float],
    factors: Sequence[DimensionedFactor],
    design: TargetFreeSensitivityDesign,
    *,
    perturbation_fraction: float | None = None,
) -> SensitivityCampaign:
    """Run a deterministic one-factor-at-a-time local derivative campaign."""
    if not factors:
        raise ValueError("at least one factor is required")
    names = [f.name for f in factors]
    if len(set(names)) != len(names):
        raise ValueError("factor names must be unique")
    nonparametric = [f.name for f in factors if f.axis is not SensitivityAxis.PARAMETRIC]
    if nonparametric:
        raise ValueError(
            "QA-046 parametric campaign cannot pool numerical/model-form axes: "
            + ", ".join(nonparametric)
        )
    base = {k: _finite(f"parameter {k}", v) for k, v in baseline_parameters.items()}
    q0 = _validate_qoi(evaluator(base), design.qoi_units)
    estimates: list[LocalSensitivityEstimate] = []
    for factor in factors:
        estimates.extend(central_local_sensitivity(
            evaluator, base, factor, design, perturbation_fraction=perturbation_fraction
        ))
    return SensitivityCampaign(design, base, q0, tuple(estimates))
