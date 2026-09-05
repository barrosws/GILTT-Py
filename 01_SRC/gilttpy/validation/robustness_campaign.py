"""QA-042 robustness diagnostics for the modern conservative transport solver.

This module is validation-only.  It does not alter production physics or the
historical branch.  It provides reusable, target-free diagnostics for four
numerically distinct questions:

1. robustness to stronger vertical gradients in u(z) and K(z),
2. invariance to translation of the explicit lower interface z_lower,
3. conservation and positivity diagnostics for singular point-source inlet data,
4. controlled replacement of the Dirac inlet by a source-tagged smooth Gaussian
   inlet profile for cross-discretization studies.

Conservation, convergence, positivity and pointwise accuracy are intentionally
reported separately; no clipping/flooring is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.basis.shifted_legendre import lower_values, values as legendre_values

FloatArray = NDArray[np.float64]
Profile = Callable[[FloatArray], ArrayLike]


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class GaussianInletProfile:
    """Smooth positive inlet profile normalized by advective throughflow.

    ``center_m`` and ``sigma_m`` are physical heights/lengths.  The raw Gaussian
    is truncated only by the declared computational interval [z_lower,h].  The
    normalization is chosen so that ``integral u(z) C(0,z) dz = source_rate``.
    The object carries explicit label/provenance so source regularization cannot
    be introduced silently.
    """

    z_lower: float
    h: float
    center_m: float
    sigma_m: float
    source_rate: float = 1.0
    label: str = "QA042 Gaussian inlet"
    provenance: str = "QA042 validation-only source regularization; no target tuning"

    def __post_init__(self) -> None:
        zl = _finite("z_lower", self.z_lower)
        h = _finite("h", self.h)
        if h <= zl:
            raise ValueError("require h > z_lower")
        c = _finite("center_m", self.center_m)
        if not zl <= c <= h:
            raise ValueError("center_m outside [z_lower,h]")
        _positive("sigma_m", self.sigma_m)
        sr = _finite("source_rate", self.source_rate)
        if sr < 0.0:
            raise ValueError("source_rate must be nonnegative")
        if not str(self.label).strip() or not str(self.provenance).strip():
            raise ValueError("label and provenance are required")

    def raw(self, z: ArrayLike) -> FloatArray:
        zz = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(zz)) or np.any(zz < self.z_lower) or np.any(zz > self.h):
            raise ValueError("z outside [z_lower,h]")
        return np.asarray(np.exp(-0.5*((zz-self.center_m)/self.sigma_m)**2), dtype=np.float64)

    def normalization(self, wind: Profile, *, n_quad: int = 1024) -> float:
        zq, wq = gauss_legendre_interval(self.z_lower, self.h, int(n_quad))
        u = np.asarray(wind(zq), dtype=np.float64)
        if u.shape != zq.shape or not np.all(np.isfinite(u)) or np.any(u <= 0.0):
            raise ValueError("wind must be finite and positive on the Gaussian source domain")
        denom = float(np.sum(wq*u*self.raw(zq)))
        if not math.isfinite(denom) or denom <= 0.0:
            raise FloatingPointError("invalid Gaussian advective normalization")
        return float(self.source_rate/denom)

    def profile(self, wind: Profile, *, n_quad: int = 1024) -> Callable[[FloatArray], FloatArray]:
        amp = self.normalization(wind, n_quad=n_quad)
        return lambda z: np.asarray(amp*self.raw(z), dtype=np.float64)


@dataclass(frozen=True)
class FiniteDimensionalBudget:
    x_end: float
    inlet_advective_flux: float
    outlet_advective_flux: float
    integrated_lower_flux: float
    residual: float

    @property
    def relative_residual(self) -> float:
        return abs(self.residual)/max(abs(self.inlet_advective_flux), np.finfo(float).tiny)


@dataclass(frozen=True)
class PositivityDiagnostic:
    minimum: float
    maximum: float
    negative_peak_ratio: float
    negative_l1_fraction: float


def propagated_coefficients(system, y0: ArrayLike, x: float) -> FloatArray:
    """Propagate an arbitrary inlet coefficient vector through a frozen QA-038 system."""
    x = _finite("x", x)
    if x < 0.0:
        raise ValueError("x must be nonnegative")
    yy = np.asarray(y0, dtype=np.float64)
    if yy.shape != (system.n_modes,) or not np.all(np.isfinite(yy)):
        raise ValueError("invalid y0")
    return np.asarray(np.real_if_close(expm(system.generator*x) @ yy, tol=1000), dtype=np.float64)


def concentration_from_coefficients(system, y: ArrayLike, z: ArrayLike) -> FloatArray:
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != (system.n_modes,) or not np.all(np.isfinite(yy)):
        raise ValueError("invalid coefficient vector")
    zz = np.asarray(z, dtype=np.float64)
    return np.asarray(
        legendre_values(zz, h=system.h, n_modes=system.n_modes, z_lower=system.z_lower) @ yy,
        dtype=np.float64,
    )


def project_profile_to_system(profile: Profile, system, *, n_quad: int = 1024) -> FloatArray:
    """L2-project a smooth inlet profile onto the system's shifted-Legendre basis."""
    zq, wq = gauss_legendre_interval(system.z_lower, system.h, int(n_quad))
    phi = legendre_values(zq, h=system.h, n_modes=system.n_modes, z_lower=system.z_lower)
    val = np.asarray(profile(zq), dtype=np.float64)
    if val.shape != zq.shape or not np.all(np.isfinite(val)):
        raise ValueError("profile must return finite values with matching shape")
    return np.asarray(phi.T @ (wq*val), dtype=np.float64)


def advective_flux_from_coefficients(system, y: ArrayLike) -> float:
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != (system.n_modes,):
        raise ValueError("coefficient shape mismatch")
    return float(math.sqrt(system.interval_length) * (system.M[0] @ yy))


def lower_flux_from_coefficients(system, y: ArrayLike) -> float:
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != (system.n_modes,):
        raise ValueError("coefficient shape mismatch")
    b = lower_values(h=system.h, n_modes=system.n_modes, z_lower=system.z_lower)
    return float(system.boundary_sink_velocity_m_s * (b @ yy))


def integrated_state(system, y0: ArrayLike, x: float) -> FloatArray:
    """Return int_0^x exp(G xi)y0 dxi without assuming G invertible."""
    x = _finite("x", x)
    if x < 0.0:
        raise ValueError("x must be nonnegative")
    yy = np.asarray(y0, dtype=np.float64)
    n = system.n_modes
    if yy.shape != (n,) or not np.all(np.isfinite(yy)):
        raise ValueError("invalid y0")
    if x == 0.0:
        return np.zeros_like(yy)
    aug = np.zeros((n+1, n+1), dtype=np.float64)
    aug[:n, :n] = system.generator
    aug[:n, n] = yy
    state = np.zeros(n+1, dtype=np.float64)
    state[n] = 1.0
    out = expm(aug*x) @ state
    return np.asarray(out[:n], dtype=np.float64)


def finite_dimensional_budget(system, y0: ArrayLike, x_end: float) -> FiniteDimensionalBudget:
    """Exact budget for an arbitrary finite-dimensional inlet state."""
    x = _finite("x_end", x_end)
    if x < 0.0:
        raise ValueError("x_end must be nonnegative")
    yy = np.asarray(y0, dtype=np.float64)
    yx = propagated_coefficients(system, yy, x)
    iy = integrated_state(system, yy, x)
    fin = advective_flux_from_coefficients(system, yy)
    fout = advective_flux_from_coefficients(system, yx)
    dep = lower_flux_from_coefficients(system, iy)
    return FiniteDimensionalBudget(x, fin, fout, dep, fin-fout-dep)


def positivity_diagnostic(values: ArrayLike, *, weights: ArrayLike | None = None) -> PositivityDiagnostic:
    """Report raw negativity without clipping or flooring."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("values must be a finite nonempty vector")
    vmax = float(np.max(v)); vmin = float(np.min(v))
    neg_peak = max(0.0, -vmin)/max(abs(vmax), np.finfo(float).tiny)
    if weights is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != v.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("invalid weights")
    neg = np.maximum(-v, 0.0)
    pos = np.maximum(v, 0.0)
    neg_l1 = float(np.sum(w*neg))
    pos_l1 = float(np.sum(w*pos))
    frac = neg_l1/max(neg_l1+pos_l1, np.finfo(float).tiny)
    return PositivityDiagnostic(vmin, vmax, float(neg_peak), float(frac))
