"""Global mass-conservation diagnostics for modern GILTT-Py transport.

QA-039 closes the source-to-domain-to-surface budget for the conservative
shifted-Legendre aerosol solver introduced in QA-038.

For steady transport on ``x in [0,X]`` and ``z in [z_lower,h]``,

    u C_x = d/dz (Kz C_z + Vg C),

with zero total flux at the top and total positive-downward lower flux
``J_down``, vertical integration gives

    d F_adv / dx = -J_down,

and therefore

    Q_in = F_adv(X) + int_0^X J_down dx.

For the zero-initial-condition transient problem, the Laplace-domain identity is

    s Ibar + d Fbar_adv/dx + Jbar_down = 0,

so over ``[0,X]``

    Q/s = Fbar_adv(X) + int_0^X Jbar_down dx + s int_0^X Ibar dx.

The module is diagnostic only.  It does not select aerosol physics, infer
surface parameters, alter boundary laws, or touch the historical branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from gilttpy.basis.shifted_legendre import lower_values
from gilttpy.solvers.settling_2d_legendre import (
    SettlingLegendreSystem,
    TransientSettlingLegendreSystem,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _nonnegative_x(x: float) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError("x must be finite and nonnegative")
    return x


def _integrated_linear_state(generator: NDArray, y0: NDArray, x: float) -> NDArray:
    """Return ``int_0^x exp(G xi) y0 dxi`` without assuming G invertible."""
    x = _nonnegative_x(x)
    g = np.asarray(generator)
    y0 = np.asarray(y0)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or y0.shape != (g.shape[0],):
        raise ValueError("generator/y0 dimension mismatch")
    if x == 0.0:
        return np.zeros_like(y0)
    n = g.shape[0]
    dtype = np.result_type(g.dtype, y0.dtype)
    aug = np.zeros((n + 1, n + 1), dtype=dtype)
    aug[:n, :n] = g
    aug[:n, n] = y0
    state0 = np.zeros(n + 1, dtype=dtype)
    state0[n] = 1.0
    out = expm(aug * x) @ state0
    return np.asarray(out[:n], dtype=dtype)


def _z_integral_vector(system: SettlingLegendreSystem) -> FloatArray:
    # Shifted Legendre basis is L2-orthonormal on [z_lower,h].  Only the
    # constant first basis function has nonzero vertical integral.
    v = np.zeros(system.n_modes, dtype=np.float64)
    v[0] = math.sqrt(system.interval_length)
    return v


def _advective_flux_from_coefficients(system: SettlingLegendreSystem, y: NDArray) -> complex:
    # With phi_0 = 1/sqrt(L), integral u C dz = sqrt(L) * row_0(M) Y.
    return complex(math.sqrt(system.interval_length) * (system.M[0].astype(y.dtype, copy=False) @ y))


def _lower_flux_from_coefficients(system: SettlingLegendreSystem, y: NDArray) -> complex:
    b = lower_values(h=system.h, n_modes=system.n_modes, z_lower=system.z_lower)
    return complex(float(system.boundary_sink_velocity_m_s) * (b.astype(y.dtype, copy=False) @ y))


def source_rate_from_inlet_condition(system: SettlingLegendreSystem) -> float:
    """Recover the inlet source rate represented by ``M y0 = Q phi(Hs)``."""
    return float(math.sqrt(system.interval_length) * system.rhs_source[0])


@dataclass(frozen=True)
class SteadyMassBudget:
    x_end: float
    source_rate: float
    inlet_advective_flux: float
    outlet_advective_flux: float
    integrated_lower_deposition: float
    weak_top_flux: float
    residual: float

    @property
    def relative_residual(self) -> float:
        return abs(self.residual) / max(abs(self.source_rate), np.finfo(float).tiny)

    @property
    def deposited_fraction(self) -> float:
        return self.integrated_lower_deposition / self.source_rate if self.source_rate != 0.0 else 0.0

    @property
    def outlet_fraction(self) -> float:
        return self.outlet_advective_flux / self.source_rate if self.source_rate != 0.0 else 0.0


def steady_mass_budget(system: SettlingLegendreSystem, x_end: float) -> SteadyMassBudget:
    """Return the exact finite-dimensional steady source/outflow/deposition budget."""
    x = _nonnegative_x(x_end)
    y0 = np.asarray(system.y0, dtype=np.float64)
    yx = np.asarray(system.coefficients(x), dtype=np.float64)
    iy = _integrated_linear_state(system.generator, y0, x)
    source = source_rate_from_inlet_condition(system)
    fin = float(np.real(_advective_flux_from_coefficients(system, y0)))
    fout = float(np.real(_advective_flux_from_coefficients(system, yx)))
    dep = float(np.real(_lower_flux_from_coefficients(system, iy)))
    # The upper boundary is the natural zero-total-flux condition in the weak
    # problem.  Do not replace it by a pointwise strong derivative diagnostic.
    top = 0.0
    residual = source - fout - dep - top
    return SteadyMassBudget(x, source, fin, fout, dep, top, residual)


@dataclass(frozen=True)
class LaplaceMassBudget:
    x_end: float
    s: complex
    source_transform: complex
    inlet_advective_flux_transform: complex
    outlet_advective_flux_transform: complex
    integrated_lower_deposition_transform: complex
    storage_rate_transform: complex
    weak_top_flux_transform: complex
    residual: complex

    @property
    def relative_residual(self) -> float:
        scale = max(abs(self.source_transform), np.finfo(float).tiny)
        return abs(self.residual) / scale


def transient_laplace_mass_budget(
    system: TransientSettlingLegendreSystem,
    x_end: float,
    s: complex,
) -> LaplaceMassBudget:
    """Return the exact finite-dimensional transient budget in Laplace space."""
    x = _nonnegative_x(x_end)
    s = system._validate_s(s)
    steady = system.steady
    y0 = np.asarray(steady.y0, dtype=np.complex128) / s
    gen = system.laplace_generator(s)
    yx = np.asarray(system.laplace_coefficients(x, s), dtype=np.complex128)
    iy = _integrated_linear_state(gen, y0, x)
    source = complex(source_rate_from_inlet_condition(steady) / s)
    fin = _advective_flux_from_coefficients(steady, y0)
    fout = _advective_flux_from_coefficients(steady, yx)
    dep = _lower_flux_from_coefficients(steady, iy)
    zint = _z_integral_vector(steady).astype(np.complex128)
    storage_rate = complex(s * (zint @ iy))
    top = 0.0j
    residual = source - fout - dep - storage_rate - top
    return LaplaceMassBudget(x, s, source, fin, fout, dep, storage_rate, top, residual)


def transient_local_laplace_residual(
    system: TransientSettlingLegendreSystem,
    x: float,
    s: complex,
) -> complex:
    """Return ``s Ibar + dFbar_adv/dx + Jbar_down`` at one x location."""
    x = _nonnegative_x(x)
    s = system._validate_s(s)
    steady = system.steady
    y = np.asarray(system.laplace_coefficients(x, s), dtype=np.complex128)
    yp = np.asarray(system.laplace_generator(s) @ y, dtype=np.complex128)
    inventory = complex(_z_integral_vector(steady).astype(np.complex128) @ y)
    dflux_dx = _advective_flux_from_coefficients(steady, yp)
    j = _lower_flux_from_coefficients(steady, y)
    return complex(s * inventory + dflux_dx + j)


@dataclass(frozen=True)
class AggregatedMassBudget:
    source_rate: float
    outlet_advective_flux: float
    integrated_lower_deposition: float
    weak_top_flux: float
    residual: float

    @property
    def relative_residual(self) -> float:
        return abs(self.residual) / max(abs(self.source_rate), np.finfo(float).tiny)


def aggregate_steady_mass_budgets(
    weighted_budgets: Iterable[tuple[float, SteadyMassBudget]],
) -> AggregatedMassBudget:
    """Aggregate independently conservative size/species budgets by linear weights.

    Weights must be finite and nonnegative.  They may be normalized fractions or
    physical source amounts; no normalization is imposed or inferred.
    """
    src = out = dep = top = 0.0
    count = 0
    for weight, budget in weighted_budgets:
        w = float(weight)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError("budget weights must be finite and nonnegative")
        src += w * budget.source_rate
        out += w * budget.outlet_advective_flux
        dep += w * budget.integrated_lower_deposition
        top += w * budget.weak_top_flux
        count += 1
    if count == 0:
        raise ValueError("at least one weighted budget is required")
    residual = src - out - dep - top
    return AggregatedMassBudget(src, out, dep, top, residual)
