"""Gaussian-type numerical inversion of Laplace transforms used by GILTT-Py.

The Buske (2008) thesis writes the inverse-Laplace approximation as

    f(t) ~= sum_k (P_k/t) A_k F(P_k/t),

and states that the Gaussian rule uses roots P_k and weights A_k tabulated
by Stroud and Secrest (1966).

For M=2, a same-lineage paper by Wendland & Vilhena (2001) reproduces the
Stroud-Secrest complex Gaussian table explicitly.  The positive-imaginary
member is

    P = 2 + i*sqrt(2),
    A = 1/2 + i*sqrt(2),

with the conjugate member required by the real-valued inversion.  This is
numerically identical to the earlier Salzer/factorial-moment reconstruction.
The constants are therefore upgraded from mathematical reconstruction to
LINEAGE_SOURCE_CONFIRMED.  Direct recovery of the literal Buske-2008 Fortran
assignment remains desirable but is no longer required to know the M=2 rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class GaussianInverseRule:
    """Abscissae and weights for ``sum (P/t) A F(P/t)``."""

    order: int
    roots: ComplexArray
    weights: ComplexArray
    rule_id: str
    provenance_class: str

    def __post_init__(self) -> None:
        p = np.asarray(self.roots, dtype=np.complex128)
        a = np.asarray(self.weights, dtype=np.complex128)
        if int(self.order) != self.order or self.order < 1:
            raise ValueError("order must be a positive integer")
        if p.ndim != 1 or a.ndim != 1 or p.shape != a.shape:
            raise ValueError("roots and weights must be one-dimensional arrays of equal shape")
        if p.size != int(self.order):
            raise ValueError("number of roots/weights must equal order")
        if np.any(~np.isfinite(p.real)) or np.any(~np.isfinite(p.imag)):
            raise ValueError("roots must be finite")
        if np.any(~np.isfinite(a.real)) or np.any(~np.isfinite(a.imag)):
            raise ValueError("weights must be finite")
        if np.any(np.abs(p) == 0.0):
            raise ValueError("roots must be nonzero")
        object.__setattr__(self, "roots", p)
        object.__setattr__(self, "weights", a)


def stroud_secrest_m2_lineage_rule() -> GaussianInverseRule:
    """Return the M=2 complex Gaussian inverse-Laplace rule.

    Wendland & Vilhena (2001) explicitly reproduce the Stroud-Secrest table:
    P = 2 + i*sqrt(2), A = 1/2 + i*sqrt(2).  The second term is its complex
    conjugate.  This is the same mathematical rule previously reconstructed
    from the Salzer factorial-moment conditions.
    """
    r2 = np.sqrt(2.0)
    roots = np.asarray([2.0 + 1j * r2, 2.0 - 1j * r2], dtype=np.complex128)
    weights = np.asarray([0.5 + 1j * r2, 0.5 - 1j * r2], dtype=np.complex128)
    return GaussianInverseRule(
        order=2,
        roots=roots,
        weights=weights,
        rule_id="stroud_secrest_complex_gaussian_m2_lineage",
        provenance_class="lineage_source_confirmed_not_literal_buske_fortran",
    )


def salzer_m2_reconstructed_rule() -> GaussianInverseRule:
    """Backward-compatible alias for the now source-confirmed M=2 rule."""
    return stroud_secrest_m2_lineage_rule()


def factorial_moment_residuals(
    rule: GaussianInverseRule, *, max_power: int | None = None
) -> ComplexArray:
    """Return residuals of ``sum A/P**m = 1/m!``.

    For a rule of order M, the Buske/Salzer exactness statement corresponds to
    powers m=0,...,2M-1 for transforms 1/s, ..., 1/s**(2M).
    """
    if max_power is None:
        max_power = 2 * int(rule.order) - 1
    if int(max_power) != max_power or max_power < 0:
        raise ValueError("max_power must be a nonnegative integer")
    residuals = []
    for m in range(int(max_power) + 1):
        lhs = np.sum(rule.weights / (rule.roots ** m))
        rhs = 1.0 / float(factorial(m))
        residuals.append(lhs - rhs)
    return np.asarray(residuals, dtype=np.complex128)


def validate_factorial_moments(
    rule: GaussianInverseRule, *, atol: float = 5e-14
) -> float:
    """Validate the exactness moments and return the maximum absolute residual."""
    residuals = factorial_moment_residuals(rule)
    err = float(np.max(np.abs(residuals)))
    if err > float(atol):
        raise FloatingPointError(
            f"inverse-Laplace quadrature moment residual {err:.3e} exceeds {atol:.3e}"
        )
    return err


def gaussian_inverse_laplace(
    laplace_fn: Callable[[complex], complex],
    t: float,
    *,
    rule: GaussianInverseRule,
    imaginary_tolerance: float = 1e-10,
) -> float:
    """Evaluate ``sum_k (P_k/t) A_k F(P_k/t)`` for a supplied rule."""
    t = float(t)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and positive")
    vals = np.asarray(
        [complex(laplace_fn(complex(pk / t))) for pk in rule.roots],
        dtype=np.complex128,
    )
    if np.any(~np.isfinite(vals.real)) or np.any(~np.isfinite(vals.imag)):
        raise FloatingPointError("Laplace transform evaluation returned non-finite values")
    result = np.sum((rule.roots / t) * rule.weights * vals)
    scale = max(1.0, abs(float(result.real)))
    if abs(float(result.imag)) > float(imaginary_tolerance) * scale:
        raise FloatingPointError(
            "Gaussian inverse returned a materially complex result; "
            "check conjugate symmetry and the supplied rule"
        )
    return float(result.real)


def gaussian_inverse_m2_lineage(
    laplace_fn: Callable[[complex], complex], t: float
) -> float:
    """Convenience wrapper for the source-confirmed M=2 complex Gaussian rule."""
    rule = stroud_secrest_m2_lineage_rule()
    validate_factorial_moments(rule)
    return gaussian_inverse_laplace(laplace_fn, t, rule=rule)


def gaussian_inverse_m2_reconstructed(
    laplace_fn: Callable[[complex], complex], t: float
) -> float:
    """Backward-compatible wrapper retained for older GILTT-Py code."""
    return gaussian_inverse_m2_lineage(laplace_fn, t)
