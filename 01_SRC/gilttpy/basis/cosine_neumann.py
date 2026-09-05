"""Cosine basis for homogeneous Neumann conditions on [0, h]."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def eigenvalues(n_modes: int, h: float) -> NDArray[np.float64]:
    """Return lambda_n = n*pi/h for n=0,...,n_modes-1."""
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and > 0")
    n = np.arange(n_modes, dtype=np.float64)
    return n * np.pi / float(h)


def values(z: ArrayLike, h: float, n_modes: int) -> NDArray[np.float64]:
    """Evaluate psi_n(z)=cos(lambda_n z).

    Returns an array of shape (n_z, n_modes).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    lam = eigenvalues(n_modes, h)
    return np.cos(np.outer(z_arr, lam))


def derivatives(z: ArrayLike, h: float, n_modes: int) -> NDArray[np.float64]:
    """Evaluate d psi_n / dz for the cosine basis."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    lam = eigenvalues(n_modes, h)
    return -np.sin(np.outer(z_arr, lam)) * lam[None, :]


def analytic_norms(n_modes: int, h: float) -> NDArray[np.float64]:
    """Return integral_0^h psi_n(z)^2 dz for the unnormalised basis."""
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and > 0")
    norms = np.full(n_modes, float(h) / 2.0, dtype=np.float64)
    norms[0] = float(h)
    return norms
