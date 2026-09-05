"""Orthonormal shifted-Legendre basis on an explicit finite interval.

The canonical interval is ``[z_lower, h]``.  ``z_lower=0`` preserves the
original QA-022 API and values exactly.  This basis is used by the modern
spectral-Galerkin verification/production branch; historical GILTT branches
retain their source-specific bases and domains unchanged.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.legendre import legvander

FloatArray = NDArray[np.float64]


def _validate(h: float, n_modes: int, z_lower: float = 0.0) -> tuple[float, int, float, float]:
    h = float(h); z_lower = float(z_lower)
    if not np.isfinite(h) or not np.isfinite(z_lower) or h <= z_lower:
        raise ValueError("require finite h > z_lower")
    if int(n_modes) != n_modes or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    length = h - z_lower
    return h, int(n_modes), z_lower, length


def values(z: ArrayLike, *, h: float, n_modes: int, z_lower: float = 0.0) -> FloatArray:
    """Evaluate the L2(z_lower,h)-orthonormal shifted Legendre basis.

    phi_n(z) = sqrt((2n+1)/H) P_n(2(z-z_lower)/H-1),
    H = h-z_lower, n=0,...,N-1.
    """
    h, n_modes, z_lower, length = _validate(h, n_modes, z_lower)
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if (not np.all(np.isfinite(z_arr)) or np.any(z_arr < z_lower)
            or np.any(z_arr > h)):
        raise ValueError("z outside [z_lower,h]")
    xi = 2.0*(z_arr-z_lower)/length - 1.0
    p = legvander(xi, n_modes-1)
    scale = np.sqrt((2.0*np.arange(n_modes, dtype=np.float64)+1.0)/length)
    return np.asarray(p*scale[None, :], dtype=np.float64)


def derivatives(z: ArrayLike, *, h: float, n_modes: int, z_lower: float = 0.0) -> FloatArray:
    """Evaluate d phi_n / dz using an exact Legendre derivative transform."""
    h, n_modes, z_lower, length = _validate(h, n_modes, z_lower)
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if (not np.all(np.isfinite(z_arr)) or np.any(z_arr < z_lower)
            or np.any(z_arr > h)):
        raise ValueError("z outside [z_lower,h]")
    xi = 2.0*(z_arr-z_lower)/length - 1.0
    p = legvander(xi, n_modes-1)
    # D[k,n] is the coefficient of P_k in dP_n/dxi.
    dcoef = np.zeros((n_modes, n_modes), dtype=np.float64)
    for n in range(1, n_modes):
        for k in range(n-1, -1, -2):
            dcoef[k, n] = 2.0*k + 1.0
    dp_dxi = p @ dcoef
    scale = np.sqrt((2.0*np.arange(n_modes, dtype=np.float64)+1.0)/length)
    return np.asarray((2.0/length)*dp_dxi*scale[None, :], dtype=np.float64)


def lower_values(*, h: float, n_modes: int, z_lower: float = 0.0) -> FloatArray:
    """phi_n(z_lower) in closed form."""
    _, n_modes, _, length = _validate(h, n_modes, z_lower)
    n = np.arange(n_modes, dtype=np.float64)
    return ((-1.0)**n)*np.sqrt((2.0*n+1.0)/length)


def ground_values(*, h: float, n_modes: int) -> FloatArray:
    """Backward-compatible alias for phi_n(0) on the interval [0,h]."""
    return lower_values(h=h, n_modes=n_modes, z_lower=0.0)


def top_values(*, h: float, n_modes: int, z_lower: float = 0.0) -> FloatArray:
    """phi_n(h) in closed form on [z_lower,h]."""
    _, n_modes, _, length = _validate(h, n_modes, z_lower)
    n = np.arange(n_modes, dtype=np.float64)
    return np.sqrt((2.0*n+1.0)/length)
