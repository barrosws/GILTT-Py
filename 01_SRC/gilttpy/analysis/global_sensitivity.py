"""Target-free variance-based global sensitivity infrastructure for GILTT-Py 2.0.

QA-048 quantifies the variance apportionment induced by *declared independent
parametric probability distributions*.  It does not infer distributions from
bounds, observations, calibration residuals, local rankings, numerical controls,
or model-form alternatives.

The implementation uses two independently scrambled Sobol' base matrices A and B
constructed from one 2D-dimensional scrambled Sobol' design.  First-order Sobol'
indices use a Saltelli-style covariance estimator; total-order indices use the
Jansen squared-difference estimator.  When requested, pairwise second-order terms
are estimated with the Saltelli A/B cross design.  All estimates remain conditional
on the declared input distributions and independence assumption.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import math
import numpy as np
from scipy.stats import qmc

from gilttpy.analysis.uncertainty import (
    ParametricUncertaintySpec,
    TargetFreeUncertaintyDesign,
    UncertaintyRepresentation,
)


Evaluator = Callable[[Mapping[str, float]], Mapping[str, float]]


def _finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _validate_power_of_two(n_base: int) -> int:
    n = int(n_base)
    if n < 8 or n & (n - 1):
        raise ValueError("Sobol QA-048 base sample count must be a power of two >= 8")
    return n


@dataclass(frozen=True)
class SobolFactorIndices:
    factor_name: str
    first_order: float
    total_order: float

    @property
    def aggregate_interaction_involvement(self) -> float:
        """Raw ST-S1 diagnostic; not a pairwise interaction index."""
        return float(self.total_order - self.first_order)


@dataclass(frozen=True)
class SobolSecondOrderIndex:
    factor_a: str
    factor_b: str
    second_order: float


@dataclass(frozen=True)
class SobolQOIResult:
    qoi_name: str
    unit: str
    output_variance: float
    factors: tuple[SobolFactorIndices, ...]
    second_order: tuple[SobolSecondOrderIndex, ...]

    def factor(self, name: str) -> SobolFactorIndices:
        for item in self.factors:
            if item.factor_name == name:
                return item
        raise KeyError(name)

    def pair(self, a: str, b: str) -> SobolSecondOrderIndex:
        key = frozenset((a, b))
        for item in self.second_order:
            if frozenset((item.factor_a, item.factor_b)) == key:
                return item
        raise KeyError((a, b))


@dataclass(frozen=True)
class GlobalSensitivityResult:
    design: TargetFreeUncertaintyDesign
    specs: tuple[ParametricUncertaintySpec, ...]
    n_base: int
    seed: int
    calc_second_order: bool
    factor_names: tuple[str, ...]
    unit_a: np.ndarray
    unit_b: np.ndarray
    qoi_a: Mapping[str, np.ndarray]
    qoi_b: Mapping[str, np.ndarray]
    qoi_ab: Mapping[str, np.ndarray]
    qoi_ba: Mapping[str, np.ndarray]
    qoi_results: tuple[SobolQOIResult, ...]

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)

    @property
    def n_model_evaluations(self) -> int:
        multiplier = 2 + self.n_factors * (2 if self.calc_second_order else 1)
        return self.n_base * multiplier

    def qoi(self, qoi_name: str) -> SobolQOIResult:
        for item in self.qoi_results:
            if item.qoi_name == qoi_name:
                return item
        raise KeyError(qoi_name)


def _validate_contract(
    specs: Sequence[ParametricUncertaintySpec],
    design: TargetFreeUncertaintyDesign,
) -> tuple[ParametricUncertaintySpec, ...]:
    if not specs:
        raise ValueError("at least one uncertainty specification is required")
    if not design.independent_inputs:
        raise ValueError("classical Sobol decomposition requires explicitly independent inputs")
    if design.observational_target_used:
        raise ValueError("QA-048 forbids observational/calibration targets")
    specs_t = tuple(specs)
    names = [s.name for s in specs_t]
    if len(set(names)) != len(names):
        raise ValueError("uncertainty input names must be unique")
    bad = [s.name for s in specs_t if s.representation is not UncertaintyRepresentation.PROBABILITY_DISTRIBUTION]
    if bad:
        raise ValueError("Sobol analysis requires explicit probability distributions: " + ", ".join(bad))
    return specs_t


def _transform_matrix(unit: np.ndarray, specs: Sequence[ParametricUncertaintySpec]) -> np.ndarray:
    x = np.empty_like(unit, dtype=np.float64)
    for j, spec in enumerate(specs):
        x[:, j] = spec.transform_unit(unit[:, j])
    return x


def _evaluate_rows(
    evaluator: Evaluator,
    rows: np.ndarray,
    names: Sequence[str],
    qoi_units: Mapping[str, str],
    *,
    workers: int,
) -> dict[str, np.ndarray]:
    expected = set(qoi_units)

    def one(row: np.ndarray) -> dict[str, float]:
        params = {name: float(row[j]) for j, name in enumerate(names)}
        result = evaluator(params)
        if set(result) != expected:
            raise ValueError("evaluator QOI keys must exactly match design qoi_units")
        return {name: _finite(f"QOI {name}", result[name]) for name in qoi_units}

    w = int(workers)
    if w < 1:
        raise ValueError("workers must be >= 1")
    if w == 1:
        evaluated = [one(row) for row in rows]
    else:
        # map preserves row order; determinism is independent of completion order.
        with ThreadPoolExecutor(max_workers=w) as pool:
            evaluated = list(pool.map(one, rows))
    return {
        name: np.asarray([row[name] for row in evaluated], dtype=np.float64)
        for name in qoi_units
    }


def _estimate_from_outputs(
    design: TargetFreeUncertaintyDesign,
    factor_names: Sequence[str],
    qoi_a: Mapping[str, np.ndarray],
    qoi_b: Mapping[str, np.ndarray],
    qoi_ab: Mapping[str, np.ndarray],
    qoi_ba: Mapping[str, np.ndarray],
    *,
    calc_second_order: bool,
) -> tuple[SobolQOIResult, ...]:
    d = len(factor_names)
    out: list[SobolQOIResult] = []
    for qoi_name, unit in design.qoi_units.items():
        ya = np.asarray(qoi_a[qoi_name], dtype=np.float64)
        yb = np.asarray(qoi_b[qoi_name], dtype=np.float64)
        if ya.shape != yb.shape or ya.ndim != 1:
            raise ValueError("A/B QOI outputs must be matching one-dimensional arrays")
        y = np.concatenate((ya, yb))
        mean = float(np.mean(y))
        variance = float(np.var(y, ddof=0))
        if not math.isfinite(variance) or variance <= np.finfo(np.float64).eps * max(1.0, float(np.mean(y*y))):
            raise ValueError(f"QOI {qoi_name} has zero or numerically degenerate variance")
        a0 = ya - mean
        b0 = yb - mean
        first: list[float] = []
        total: list[float] = []
        factors: list[SobolFactorIndices] = []
        for i, factor_name in enumerate(factor_names):
            yab = np.asarray(qoi_ab[qoi_name][i], dtype=np.float64) - mean
            # Saltelli-style first-order covariance estimator and Jansen total-order estimator.
            si = float(np.mean(b0 * (yab - a0)) / variance)
            sti = float(np.mean((a0 - yab)**2) / (2.0 * variance))
            first.append(si); total.append(sti)
            factors.append(SobolFactorIndices(factor_name, si, sti))

        pairs: list[SobolSecondOrderIndex] = []
        if calc_second_order:
            for j in range(d):
                yba_j = np.asarray(qoi_ba[qoi_name][j], dtype=np.float64) - mean
                for k in range(j + 1, d):
                    yab_k = np.asarray(qoi_ab[qoi_name][k], dtype=np.float64) - mean
                    closed_jk = float(np.mean(yba_j * yab_k - a0 * b0) / variance)
                    s2 = float(closed_jk - first[j] - first[k])
                    pairs.append(SobolSecondOrderIndex(factor_names[j], factor_names[k], s2))
        out.append(SobolQOIResult(qoi_name, unit, variance, tuple(factors), tuple(pairs)))
    return tuple(out)


def variance_based_global_sensitivity(
    evaluator: Evaluator,
    specs: Sequence[ParametricUncertaintySpec],
    design: TargetFreeUncertaintyDesign,
    *,
    n_base: int = 32,
    seed: int = 4801,
    calc_second_order: bool = True,
    workers: int = 1,
) -> GlobalSensitivityResult:
    """Estimate Sobol' first/total and optional pairwise second-order indices.

    Computational cost is N*(D+2) model evaluations without pairwise second-order
    terms and N*(2D+2) when second-order terms are requested.
    """
    n = _validate_power_of_two(n_base)
    specs_t = _validate_contract(specs, design)
    names = tuple(s.name for s in specs_t)
    d = len(specs_t)

    sampler = qmc.Sobol(d=2*d, scramble=True, seed=int(seed))
    base = np.asarray(sampler.random_base2(int(math.log2(n))), dtype=np.float64)
    ua, ub = base[:, :d], base[:, d:]
    xa, xb = _transform_matrix(ua, specs_t), _transform_matrix(ub, specs_t)

    matrices: list[np.ndarray] = [xa, xb]
    labels: list[tuple[str, int | None]] = [("a", None), ("b", None)]
    for i in range(d):
        x = xa.copy(); x[:, i] = xb[:, i]
        matrices.append(x); labels.append(("ab", i))
    if calc_second_order:
        for i in range(d):
            x = xb.copy(); x[:, i] = xa[:, i]
            matrices.append(x); labels.append(("ba", i))

    all_rows = np.vstack(matrices)
    evaluated = _evaluate_rows(evaluator, all_rows, names, design.qoi_units, workers=workers)
    offsets = np.cumsum([0] + [len(m) for m in matrices])

    qoi_a: dict[str, np.ndarray] = {}
    qoi_b: dict[str, np.ndarray] = {}
    qoi_ab = {name: np.empty((d, n), dtype=np.float64) for name in design.qoi_units}
    qoi_ba = {name: np.empty((d, n), dtype=np.float64) for name in design.qoi_units} if calc_second_order else {}
    for block, (kind, idx) in enumerate(labels):
        lo, hi = offsets[block], offsets[block+1]
        for qoi_name in design.qoi_units:
            arr = evaluated[qoi_name][lo:hi]
            if kind == "a": qoi_a[qoi_name] = arr
            elif kind == "b": qoi_b[qoi_name] = arr
            elif kind == "ab": qoi_ab[qoi_name][int(idx), :] = arr
            elif kind == "ba": qoi_ba[qoi_name][int(idx), :] = arr

    qoi_results = _estimate_from_outputs(
        design, names, qoi_a, qoi_b, qoi_ab, qoi_ba,
        calc_second_order=bool(calc_second_order),
    )
    return GlobalSensitivityResult(
        design=design, specs=specs_t, n_base=n, seed=int(seed),
        calc_second_order=bool(calc_second_order), factor_names=names,
        unit_a=ua, unit_b=ub, qoi_a=qoi_a, qoi_b=qoi_b,
        qoi_ab=qoi_ab, qoi_ba=qoi_ba, qoi_results=qoi_results,
    )


def reestimate_prefix(result: GlobalSensitivityResult, n_prefix: int) -> tuple[SobolQOIResult, ...]:
    """Re-estimate indices from a nested power-of-two prefix without new model calls."""
    n = _validate_power_of_two(n_prefix)
    if n > result.n_base:
        raise ValueError("prefix cannot exceed completed base sample")
    qoi_a = {k: v[:n] for k, v in result.qoi_a.items()}
    qoi_b = {k: v[:n] for k, v in result.qoi_b.items()}
    qoi_ab = {k: v[:, :n] for k, v in result.qoi_ab.items()}
    qoi_ba = {k: v[:, :n] for k, v in result.qoi_ba.items()} if result.calc_second_order else {}
    return _estimate_from_outputs(
        result.design, result.factor_names, qoi_a, qoi_b, qoi_ab, qoi_ba,
        calc_second_order=result.calc_second_order,
    )
