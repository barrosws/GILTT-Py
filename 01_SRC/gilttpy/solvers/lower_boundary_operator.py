"""Typed lower-boundary flux operators for the modern GILTT-Py solvers.

QA-037 promotes the lower-interface law from an overloaded scalar
``deposition_velocity`` into an explicit weak-boundary contract.  For a
linear Robin/compensation law

    J_down = k * (C_lower - C_eq),

the Galerkin boundary contribution is

    B = k b b^T,
    f = k C_eq b,

so the steady modal problem is

    M Y' + (S + B)Y = f.

The protocol is intentionally structural.  The physics-side
``LinearResolvedInterfaceFluxLaw`` frozen in QA-036 therefore satisfies it
without the solver importing the physics package, preserving layer
separation and avoiding a circular dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BoundaryWeakTerms:
    """Finite-dimensional weak terms contributed by one lower-boundary law."""

    matrix: FloatArray
    forcing: FloatArray

    def __post_init__(self) -> None:
        B = np.asarray(self.matrix, dtype=np.float64)
        f = np.asarray(self.forcing, dtype=np.float64)
        if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] < 1:
            raise ValueError("matrix must be a nonempty square array")
        if f.ndim != 1 or f.shape[0] != B.shape[0]:
            raise ValueError("forcing must be a vector compatible with matrix")
        if not np.all(np.isfinite(B)) or not np.all(np.isfinite(f)):
            raise ValueError("boundary weak terms must be finite")
        object.__setattr__(self, "matrix", B)
        object.__setattr__(self, "forcing", f)


@runtime_checkable
class LowerBoundaryFluxLaw(Protocol):
    """Structural contract consumed by the modern weak solvers."""

    sink_velocity_m_s: float
    equilibrium_concentration: float
    label: str

    def weak_terms(self, lower_basis_values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return ``(B,f)`` for ``M Y' + S Y + B Y = f``."""
        ...

    def downward_flux(self, concentration_at_interface: float) -> float:
        """Return positive-downward boundary flux."""
        ...


@dataclass(frozen=True)
class LinearRobinBoundaryCondition:
    """Solver-native affine Robin condition ``J_down=k(C-C_eq)``.

    This is a numerical/solver contract, not a land-surface parameterization.
    Physics providers may satisfy :class:`LowerBoundaryFluxLaw` directly.
    """

    sink_velocity_m_s: float
    equilibrium_concentration: float = 0.0
    label: str = "linear Robin lower boundary"

    def __post_init__(self) -> None:
        k = float(self.sink_velocity_m_s)
        ceq = float(self.equilibrium_concentration)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("sink_velocity_m_s must be finite and nonnegative")
        if not math.isfinite(ceq) or ceq < 0.0:
            raise ValueError("equilibrium_concentration must be finite and nonnegative")
        if k == 0.0 and ceq != 0.0:
            raise ValueError("zero sink velocity requires zero equilibrium concentration")
        if not str(self.label).strip():
            raise ValueError("label must be nonempty")

    def weak_terms(self, lower_basis_values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        b = np.asarray(lower_basis_values, dtype=np.float64)
        if b.ndim != 1 or b.size < 1 or not np.all(np.isfinite(b)):
            raise ValueError("lower_basis_values must be a finite one-dimensional vector")
        k = float(self.sink_velocity_m_s)
        B = k * np.outer(b, b)
        f = k * float(self.equilibrium_concentration) * b
        return np.asarray(B, dtype=np.float64), np.asarray(f, dtype=np.float64)

    def weak_terms_object(self, lower_basis_values: ArrayLike) -> BoundaryWeakTerms:
        B, f = self.weak_terms(lower_basis_values)
        return BoundaryWeakTerms(B, f)

    def downward_flux(self, concentration_at_interface: float) -> float:
        c = float(concentration_at_interface)
        if not math.isfinite(c) or c < 0.0:
            raise ValueError("concentration_at_interface must be finite and nonnegative")
        return float(self.sink_velocity_m_s) * (c - float(self.equilibrium_concentration))


def coerce_boundary_weak_terms(
    boundary: LowerBoundaryFluxLaw,
    lower_basis_values: ArrayLike,
) -> BoundaryWeakTerms:
    """Validate a structural boundary provider and return typed weak terms."""
    if not isinstance(boundary, LowerBoundaryFluxLaw):
        raise TypeError("boundary must satisfy LowerBoundaryFluxLaw")
    k = float(boundary.sink_velocity_m_s)
    ceq = float(boundary.equilibrium_concentration)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("boundary sink velocity must be finite and nonnegative")
    if not math.isfinite(ceq) or ceq < 0.0:
        raise ValueError("boundary equilibrium concentration must be finite and nonnegative")
    if k == 0.0 and ceq != 0.0:
        raise ValueError("zero sink velocity requires zero equilibrium concentration")
    B, f = boundary.weak_terms(lower_basis_values)
    terms = BoundaryWeakTerms(B, f)
    n = terms.matrix.shape[0]
    b = np.asarray(lower_basis_values, dtype=np.float64)
    expected_B = k * np.outer(b, b)
    expected_f = k * ceq * b
    scale_B = max(1.0, float(np.max(np.abs(expected_B))))
    scale_f = max(1.0, float(np.max(np.abs(expected_f))))
    tol = 256.0 * np.finfo(float).eps
    if not np.allclose(terms.matrix, expected_B, rtol=tol, atol=tol * scale_B):
        raise ValueError("boundary provider matrix is inconsistent with declared linear Robin law")
    if not np.allclose(terms.forcing, expected_f, rtol=tol, atol=tol * scale_f):
        raise ValueError("boundary provider forcing is inconsistent with declared linear Robin law")
    if b.size != n:
        raise ValueError("boundary weak-term dimension mismatch")
    return terms
