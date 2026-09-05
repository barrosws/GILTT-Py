"""Quadrature utilities used by GILTT-Py."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss


def gauss_legendre_interval(
    a: float, b: float, order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Legendre nodes and weights mapped from [-1,1] to [a,b]."""
    if order < 1:
        raise ValueError("order must be >= 1")
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("a and b must be finite")
    if not b > a:
        raise ValueError("require b > a")
    xi, wi = leggauss(order)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x.astype(np.float64), w.astype(np.float64)
