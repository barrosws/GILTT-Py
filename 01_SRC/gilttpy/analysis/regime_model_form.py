"""Target-free structural regime/model-form comparison for GILTT-Py 2.0.

This module deliberately treats regimes and model forms as *discrete structural
alternatives*, not as random variables.  It therefore reports deterministic
contrasts and envelopes and refuses observational targets, probability weights,
or automatic winner selection.  Probabilistic model averaging would require an
independently justified probability model and is outside QA-049.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence
import math


class StructuralAxis(str, Enum):
    REGIME = "regime"
    MODEL_FORM = "model_form"


def _text(name: str, value: str) -> str:
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
class StructuralAlternative:
    key: str
    label: str
    axis: StructuralAxis
    provenance: str

    def __post_init__(self) -> None:
        _text("key", self.key); _text("label", self.label); _text("provenance", self.provenance)
        if not isinstance(self.axis, StructuralAxis):
            raise TypeError("axis must be a StructuralAxis")


@dataclass(frozen=True)
class StructuralComparisonDesign:
    label: str
    provenance: str
    qoi_units: Mapping[str, str]
    regimes: tuple[StructuralAlternative, ...]
    model_forms: tuple[StructuralAlternative, ...]
    observational_target_used: bool = False
    probabilistic_weights_used: bool = False
    winner_selection_requested: bool = False

    def __post_init__(self) -> None:
        _text("label", self.label); _text("provenance", self.provenance)
        if self.observational_target_used:
            raise ValueError("QA-049 forbids observational/historical targets in structural comparison")
        if self.probabilistic_weights_used:
            raise ValueError("QA-049 has no justified probabilities for regimes or model forms")
        if self.winner_selection_requested:
            raise ValueError("QA-049 compares model forms but does not select a universal winner")
        if len(self.regimes) < 2 or any(x.axis is not StructuralAxis.REGIME for x in self.regimes):
            raise ValueError("at least two REGIME alternatives are required")
        if len(self.model_forms) < 2 or any(x.axis is not StructuralAxis.MODEL_FORM for x in self.model_forms):
            raise ValueError("at least two MODEL_FORM alternatives are required")
        keys = [x.key for x in self.regimes + self.model_forms]
        if len(set(keys)) != len(keys):
            raise ValueError("structural alternative keys must be globally unique")
        if not self.qoi_units:
            raise ValueError("at least one QOI is required")
        for name, unit in self.qoi_units.items():
            _text("qoi name", name); _text("qoi unit", unit)


@dataclass(frozen=True)
class StructuralEvaluation:
    regime_key: str
    model_form_key: str
    qois: Mapping[str, float]


@dataclass(frozen=True)
class ModelFormContrast:
    regime_key: str
    qoi_name: str
    model_form_a: str
    model_form_b: str
    value_a: float
    value_b: float
    absolute_difference_b_minus_a: float
    symmetric_relative_difference_b_minus_a: float


@dataclass(frozen=True)
class RegimeEnvelope:
    model_form_key: str
    qoi_name: str
    minimum: float
    maximum: float
    span: float
    relative_span_to_max_abs: float


@dataclass(frozen=True)
class StructuralComparisonResult:
    design: StructuralComparisonDesign
    evaluations: tuple[StructuralEvaluation, ...]
    model_form_contrasts: tuple[ModelFormContrast, ...]
    regime_envelopes: tuple[RegimeEnvelope, ...]

    def evaluation(self, regime_key: str, model_form_key: str) -> StructuralEvaluation:
        for row in self.evaluations:
            if row.regime_key == regime_key and row.model_form_key == model_form_key:
                return row
        raise KeyError((regime_key, model_form_key))

    def contrast(self, regime_key: str, qoi_name: str) -> ModelFormContrast:
        rows = [x for x in self.model_form_contrasts if x.regime_key == regime_key and x.qoi_name == qoi_name]
        if len(rows) != 1:
            raise KeyError((regime_key, qoi_name))
        return rows[0]


Evaluator = Callable[[StructuralAlternative, StructuralAlternative], Mapping[str, float]]


def _symmetric_relative_difference(a: float, b: float) -> float:
    den = abs(a) + abs(b)
    return 0.0 if den == 0.0 else 2.0*(b-a)/den


def compare_structural_alternatives(
    evaluator: Evaluator,
    design: StructuralComparisonDesign,
) -> StructuralComparisonResult:
    """Evaluate every declared regime/model-form pair without probabilities."""
    expected = set(design.qoi_units)
    rows: list[StructuralEvaluation] = []
    for regime in design.regimes:
        for model in design.model_forms:
            raw = evaluator(regime, model)
            if set(raw) != expected:
                raise ValueError("evaluator QOI keys must exactly match design.qoi_units")
            qois = {name: _finite(f"QOI {name}", raw[name]) for name in design.qoi_units}
            rows.append(StructuralEvaluation(regime.key, model.key, qois))

    contrasts: list[ModelFormContrast] = []
    if len(design.model_forms) == 2:
        ma, mb = design.model_forms
        lookup = {(r.regime_key, r.model_form_key): r for r in rows}
        for regime in design.regimes:
            a = lookup[(regime.key, ma.key)]
            b = lookup[(regime.key, mb.key)]
            for qoi in design.qoi_units:
                va, vb = a.qois[qoi], b.qois[qoi]
                contrasts.append(ModelFormContrast(
                    regime_key=regime.key, qoi_name=qoi,
                    model_form_a=ma.key, model_form_b=mb.key,
                    value_a=va, value_b=vb,
                    absolute_difference_b_minus_a=vb-va,
                    symmetric_relative_difference_b_minus_a=_symmetric_relative_difference(va, vb),
                ))
    else:
        raise ValueError("QA-049 v1 requires exactly two model forms for auditable pairwise contrasts")

    envelopes: list[RegimeEnvelope] = []
    for model in design.model_forms:
        for qoi in design.qoi_units:
            vals = [r.qois[qoi] for r in rows if r.model_form_key == model.key]
            lo, hi = min(vals), max(vals)
            scale = max(abs(lo), abs(hi))
            envelopes.append(RegimeEnvelope(
                model_form_key=model.key, qoi_name=qoi,
                minimum=lo, maximum=hi, span=hi-lo,
                relative_span_to_max_abs=0.0 if scale == 0.0 else (hi-lo)/scale,
            ))
    return StructuralComparisonResult(design, tuple(rows), tuple(contrasts), tuple(envelopes))
