"""Target-free parametric uncertainty propagation infrastructure for GILTT-Py 2.0.

QA-047 propagates explicitly declared *parametric* probability distributions through
model quantities of interest. It does not infer probability distributions from
bounds, calibration residuals, historical targets, or local sensitivity rankings.
Numerical and model-form uncertainty remain separate axes.

The current engine uses randomized Sobol quasi-Monte Carlo sampling with an
explicit seed and power-of-two sample count. Every input marginal carries units,
provenance, an uncertainty interpretation, and a representation type. Interval-only
knowledge is deliberately rejected by probabilistic propagation rather than being
silently converted to a uniform distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence
import math
import numpy as np
from scipy.stats import qmc, norm

from gilttpy.analysis.sensitivity import SensitivityAxis


class UncertaintyInterpretation(str, Enum):
    ALEATORY = "aleatory"
    EPISTEMIC_PROBABILISTIC = "epistemic_probabilistic"
    QA_REFERENCE = "qa_reference"


class UncertaintyRepresentation(str, Enum):
    PROBABILITY_DISTRIBUTION = "probability_distribution"
    INTERVAL_ONLY = "interval_only"


class DistributionFamily(str, Enum):
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"


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
class ParametricUncertaintySpec:
    """One physical input uncertainty specification with auditable provenance."""

    name: str
    unit: str
    baseline: float
    axis: SensitivityAxis
    representation: UncertaintyRepresentation
    interpretation: UncertaintyInterpretation
    provenance: str
    family: DistributionFamily | None = None
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _nonempty("name", self.name)
        _nonempty("unit", self.unit)
        _nonempty("provenance", self.provenance)
        baseline = _finite("baseline", self.baseline)
        if self.axis is not SensitivityAxis.PARAMETRIC:
            raise ValueError("QA-047 uncertainty propagation is restricted to PARAMETRIC inputs")
        if baseline <= 0.0:
            raise ValueError("QA-047 multiplicative physical baselines must be positive")
        if not isinstance(self.representation, UncertaintyRepresentation):
            raise TypeError("representation must be an UncertaintyRepresentation")
        if not isinstance(self.interpretation, UncertaintyInterpretation):
            raise TypeError("interpretation must be an UncertaintyInterpretation")
        if self.representation is UncertaintyRepresentation.INTERVAL_ONLY:
            if self.family is not None:
                raise ValueError("interval-only uncertainty cannot carry a probability family")
            if len(self.parameters) != 2:
                raise ValueError("interval-only uncertainty requires (lower, upper)")
            lo, hi = map(float, self.parameters)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError("invalid interval-only bounds")
            return
        if not isinstance(self.family, DistributionFamily):
            raise ValueError("probabilistic uncertainty requires an explicit distribution family")
        p = tuple(_finite("distribution parameter", x) for x in self.parameters)
        if self.family is DistributionFamily.UNIFORM:
            if len(p) != 2 or not p[0] < p[1]:
                raise ValueError("uniform requires lower < upper")
        elif self.family is DistributionFamily.TRIANGULAR:
            if len(p) != 3 or not p[0] <= p[1] <= p[2] or p[0] == p[2]:
                raise ValueError("triangular requires lower <= mode <= upper with nonzero width")
        elif self.family is DistributionFamily.LOGNORMAL:
            if len(p) != 2 or p[0] <= 0.0 or p[1] <= 1.0:
                raise ValueError("lognormal requires median>0 and geometric_sd>1")

    def transform_unit(self, u: np.ndarray) -> np.ndarray:
        """Transform U(0,1) variates to this marginal distribution."""
        if self.representation is not UncertaintyRepresentation.PROBABILITY_DISTRIBUTION:
            raise ValueError(
                f"{self.name}: interval-only knowledge cannot be silently probabilized"
            )
        uu = np.asarray(u, dtype=np.float64)
        if np.any((uu < 0.0) | (uu > 1.0)):
            raise ValueError("unit variates must lie in [0,1]")
        p = tuple(float(x) for x in self.parameters)
        if self.family is DistributionFamily.UNIFORM:
            return p[0] + (p[1]-p[0])*uu
        if self.family is DistributionFamily.TRIANGULAR:
            lo, mode, hi = p
            c = (mode-lo)/(hi-lo)
            left = lo + np.sqrt(uu*(hi-lo)*(mode-lo))
            right = hi - np.sqrt((1.0-uu)*(hi-lo)*(hi-mode))
            return np.where(uu < c, left, right)
        if self.family is DistributionFamily.LOGNORMAL:
            median, gsd = p
            eps = np.finfo(np.float64).eps
            z = norm.ppf(np.clip(uu, eps, 1.0-eps))
            return median*np.exp(np.log(gsd)*z)
        raise AssertionError("unreachable distribution family")

    def analytic_mean_variance(self) -> tuple[float, float]:
        """Exact marginal mean/variance for verification and delta-method diagnostics."""
        if self.representation is not UncertaintyRepresentation.PROBABILITY_DISTRIBUTION:
            raise ValueError("interval-only uncertainty has no declared probability moments")
        p = tuple(float(x) for x in self.parameters)
        if self.family is DistributionFamily.UNIFORM:
            lo, hi = p
            return 0.5*(lo+hi), (hi-lo)**2/12.0
        if self.family is DistributionFamily.TRIANGULAR:
            lo, mode, hi = p
            mean = (lo+mode+hi)/3.0
            var = (lo*lo+mode*mode+hi*hi-lo*mode-lo*hi-mode*hi)/18.0
            return mean, var
        if self.family is DistributionFamily.LOGNORMAL:
            median, gsd = p
            sigma = math.log(gsd)
            mean = median*math.exp(0.5*sigma*sigma)
            var = (math.exp(sigma*sigma)-1.0)*math.exp(2.0*math.log(median)+sigma*sigma)
            return mean, var
        raise AssertionError("unreachable distribution family")


@dataclass(frozen=True)
class TargetFreeUncertaintyDesign:
    label: str
    provenance: str
    qoi_units: Mapping[str, str]
    independent_inputs: bool
    dependence_provenance: str
    observational_target_used: bool = False

    def __post_init__(self) -> None:
        _nonempty("label", self.label)
        _nonempty("provenance", self.provenance)
        _nonempty("dependence_provenance", self.dependence_provenance)
        if self.observational_target_used:
            raise ValueError("QA-047 forbids observational/calibration targets")
        if not self.independent_inputs:
            raise ValueError(
                "QA-047 v1 requires explicitly justified independent inputs; correlated-input "
                "propagation remains a separate HOLD rather than an implicit assumption"
            )
        if not self.qoi_units:
            raise ValueError("at least one QOI is required")
        for name, unit in self.qoi_units.items():
            _nonempty("qoi name", name); _nonempty("qoi unit", unit)


@dataclass(frozen=True)
class QOISummary:
    qoi_name: str
    unit: str
    mean: float
    standard_deviation: float
    q025: float
    median: float
    q975: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class UncertaintyPropagationResult:
    design: TargetFreeUncertaintyDesign
    specs: tuple[ParametricUncertaintySpec, ...]
    n_samples: int
    seed: int
    samples: Mapping[str, np.ndarray]
    qoi_samples: Mapping[str, np.ndarray]
    summaries: tuple[QOISummary, ...]

    def summary(self, qoi_name: str) -> QOISummary:
        for s in self.summaries:
            if s.qoi_name == qoi_name:
                return s
        raise KeyError(qoi_name)


Evaluator = Callable[[Mapping[str, float]], Mapping[str, float]]


def _validate_power_of_two(n_samples: int) -> int:
    n = int(n_samples)
    if n < 8 or n & (n-1):
        raise ValueError("Sobol QA-047 sample count must be a power of two >= 8")
    return n


def propagate_parametric_uncertainty(
    evaluator: Evaluator,
    specs: Sequence[ParametricUncertaintySpec],
    design: TargetFreeUncertaintyDesign,
    *,
    n_samples: int = 64,
    seed: int = 4701,
) -> UncertaintyPropagationResult:
    """Propagate declared independent probability marginals by randomized Sobol QMC."""
    n = _validate_power_of_two(n_samples)
    if not specs:
        raise ValueError("at least one uncertainty specification is required")
    specs_t = tuple(specs)
    names = [s.name for s in specs_t]
    if len(set(names)) != len(names):
        raise ValueError("uncertainty input names must be unique")
    interval_only = [s.name for s in specs_t if s.representation is UncertaintyRepresentation.INTERVAL_ONLY]
    if interval_only:
        raise ValueError(
            "interval-only uncertainty cannot enter probabilistic propagation: " + ", ".join(interval_only)
        )

    sampler = qmc.Sobol(d=len(specs_t), scramble=True, seed=int(seed))
    u = sampler.random_base2(int(math.log2(n)))
    sample_map = {spec.name: spec.transform_unit(u[:, j]) for j, spec in enumerate(specs_t)}

    qoi_rows: list[dict[str, float]] = []
    expected = set(design.qoi_units)
    for i in range(n):
        params = {name: float(sample_map[name][i]) for name in names}
        result = evaluator(params)
        if set(result) != expected:
            raise ValueError("evaluator QOI keys must exactly match design qoi_units")
        row = {k: _finite(f"QOI {k}", result[k]) for k in design.qoi_units}
        qoi_rows.append(row)

    qoi_samples = {
        name: np.asarray([row[name] for row in qoi_rows], dtype=np.float64)
        for name in design.qoi_units
    }
    summaries = []
    for name, unit in design.qoi_units.items():
        arr = qoi_samples[name]
        q025, med, q975 = np.quantile(arr, [0.025, 0.5, 0.975], method="linear")
        summaries.append(QOISummary(
            qoi_name=name,
            unit=unit,
            mean=float(np.mean(arr)),
            standard_deviation=float(np.std(arr, ddof=1)),
            q025=float(q025), median=float(med), q975=float(q975),
            minimum=float(np.min(arr)), maximum=float(np.max(arr)),
        ))
    return UncertaintyPropagationResult(
        design=design, specs=specs_t, n_samples=n, seed=int(seed),
        samples=sample_map, qoi_samples=qoi_samples, summaries=tuple(summaries),
    )
