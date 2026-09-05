"""Modern shifted-Legendre transport with resolved gravitational settling.

QA-038 solves, on the explicit modern interval ``[z_lower,h]``,

    u C_x = d/dz (Kz C_z + Vg C),

where ``z`` is positive upward and ``Vg >= 0`` is a downward terminal settling
speed.  Thus the positive-downward vertical flux is

    F_down = Kz C_z + Vg C.

The top boundary is the natural zero-total-flux condition.  The lower boundary
is supplied by the typed QA-037 contract and represents the **total** downward
flux leaving the resolved domain.  QA-038 restricts this settling path to a
homogeneous aerosol boundary (C_eq=0); re-emission is not mixed with particle
settling in this gate.

The weak steady system is

    M Y_x + (S + W + B)Y = 0,

    W_ij = int Vg phi_j d(phi_i)/dz dz,

with ``B`` supplied by the total interface-flux law.  ``W`` is nonsymmetric, so
this module uses a general matrix exponential rather than the symmetric
``eigh`` path used by the no-settling QA-037 operator.  At ``Vg=0`` the system
reduces exactly to the QA-037 homogeneous solver.

For the temporal Laplace problem with zero initial concentration,

    M Y_x + (S + W + B + sG)Y = 0,
    M Y(0,s) = (Q/s) phi(Hs),

and the orthonormal shifted-Legendre basis gives ``G=I`` exactly.

Historical Buske/Ribes branches are not imported or modified by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, solve

from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.basis.shifted_legendre import values as legendre_values
from gilttpy.basis.shifted_legendre import derivatives as legendre_derivatives
from gilttpy.basis.shifted_legendre import lower_values, top_values
from gilttpy.solvers.lower_boundary_operator import LowerBoundaryFluxLaw, coerce_boundary_weak_terms
from gilttpy.solvers.steady_2d_deposition_legendre import _assemble_interior
from gilttpy.solvers.transient_2d import fixed_talbot_inverse
from gilttpy.numerics.inverse_laplace_modern import (
    dehoog_inverse_laplace,
    dehoog_consensus_inverse_laplace,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Profile = Callable[[FloatArray], ArrayLike]


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class SettlingLegendreSystem:
    z_lower: float
    h: float
    n_modes: int
    settling_velocity_m_s: float
    boundary_sink_velocity_m_s: float
    boundary_label: str
    M: FloatArray
    diffusion_operator: FloatArray
    settling_operator: FloatArray
    boundary_matrix: FloatArray
    operator: FloatArray
    rhs_source: FloatArray
    y0: FloatArray

    @property
    def interval_length(self) -> float:
        return float(self.h - self.z_lower)

    @property
    def generator(self) -> FloatArray:
        # Preserve the QA-037 homogeneous numerical path bit-for-bit when Vg=0.
        assume = "pos" if self.settling_velocity_m_s == 0.0 else "gen"
        return np.asarray(-solve(self.M, self.operator, assume_a=assume, check_finite=True), dtype=np.float64)

    def coefficients(self, x: float) -> FloatArray:
        x = float(x)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("x must be finite and nonnegative")
        return np.asarray(np.real_if_close(expm(self.generator*x) @ self.y0, tol=1000), dtype=np.float64)

    def coefficients_derivative(self, x: float) -> FloatArray:
        return np.asarray(self.generator @ self.coefficients(x), dtype=np.float64)

    def concentration(self, x: float, z: ArrayLike) -> FloatArray:
        z_arr = np.asarray(z, dtype=np.float64)
        return legendre_values(z_arr, h=self.h, n_modes=self.n_modes, z_lower=self.z_lower) @ self.coefficients(x)

    def concentration_gradient(self, x: float, z: ArrayLike) -> FloatArray:
        z_arr = np.asarray(z, dtype=np.float64)
        return legendre_derivatives(z_arr, h=self.h, n_modes=self.n_modes, z_lower=self.z_lower) @ self.coefficients(x)

    def lower_boundary_concentration(self, x: float) -> float:
        b = lower_values(h=self.h, n_modes=self.n_modes, z_lower=self.z_lower)
        return float(b @ self.coefficients(x))

    def lower_boundary_downward_flux(self, x: float) -> float:
        return float(self.boundary_sink_velocity_m_s) * self.lower_boundary_concentration(x)

    def advective_flux(self, x: float, *, wind: Profile, n_quad: int = 512) -> float:
        zq, wq = gauss_legendre_interval(self.z_lower, self.h, int(n_quad))
        u = np.asarray(wind(zq), dtype=np.float64)
        return float(np.sum(wq * u * self.concentration(x, zq)))

    def flux_derivative_from_weak_constant_test(self, x: float) -> float:
        yp = self.coefficients_derivative(x)
        return float(np.sqrt(self.interval_length) * (self.M[0] @ yp))

    def downward_vertical_flux(self, x: float, z: ArrayLike, *, diffusivity: Profile) -> FloatArray:
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        kz = np.asarray(diffusivity(z_arr), dtype=np.float64)
        if kz.shape != z_arr.shape or not np.all(np.isfinite(kz)) or np.any(kz < 0.0):
            raise ValueError("diffusivity must return finite nonnegative values with matching shape")
        return kz * self.concentration_gradient(x, z_arr) + float(self.settling_velocity_m_s) * self.concentration(x, z_arr)


@dataclass(frozen=True)
class TransientSettlingLegendreSystem:
    steady: SettlingLegendreSystem
    G: FloatArray

    @property
    def z_lower(self) -> float:
        return self.steady.z_lower

    @property
    def h(self) -> float:
        return self.steady.h

    @property
    def n_modes(self) -> int:
        return self.steady.n_modes

    @property
    def settling_velocity_m_s(self) -> float:
        return self.steady.settling_velocity_m_s

    def _validate_s(self, s: complex) -> complex:
        s = complex(s)
        if not np.isfinite(s.real) or not np.isfinite(s.imag) or abs(s) == 0.0:
            raise ValueError("s must be finite and nonzero")
        return s

    def laplace_system_matrix(self, s: complex) -> ComplexArray:
        s = self._validate_s(s)
        return np.asarray(self.steady.operator.astype(complex) + s*self.G.astype(complex), dtype=np.complex128)

    def laplace_generator(self, s: complex) -> ComplexArray:
        s = self._validate_s(s)
        return -solve(self.steady.M.astype(complex), self.laplace_system_matrix(s), assume_a="gen", check_finite=True)

    def laplace_coefficients(self, x: float, s: complex) -> ComplexArray:
        x = float(x); s = self._validate_s(s)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("x must be finite and nonnegative")
        y0 = self.steady.y0.astype(complex) / s
        return expm(self.laplace_generator(s)*x) @ y0

    def laplace_concentration(self, x: float, z: float, s: complex) -> complex:
        z = float(z)
        if not math.isfinite(z) or z < self.z_lower or z > self.h:
            raise ValueError("z outside [z_lower,h]")
        phi = legendre_values(np.asarray([z]), h=self.h, n_modes=self.n_modes, z_lower=self.z_lower)[0]
        return complex(phi @ self.laplace_coefficients(x, s))

    def concentration_fixed_talbot(self, x: float, z: float, t: float, *, mstar: int = 9) -> float:
        return fixed_talbot_inverse(lambda s: self.laplace_concentration(x, z, s), t, mstar=mstar)

    def concentration_dehoog(self, x: float, z: float, t: float, *, degree: int = 28, working_dps: int = 40) -> float:
        """Modern single-degree de Hoog path; Fixed Talbot remains a historical comparator."""
        return dehoog_inverse_laplace(
            lambda s: self.laplace_concentration(x, z, s),
            t, degree=degree, working_dps=working_dps,
        )

    def concentration_dehoog_consensus(
        self,
        x: float,
        z: float,
        t: float,
        *,
        degrees: tuple[int, ...] = (24, 26, 28),
        working_dps: int = 40,
    ) -> float:
        """Portable degree-consensus path for complex128 spectral evaluations."""
        return dehoog_consensus_inverse_laplace(
            lambda s: self.laplace_concentration(x, z, s),
            t,
            degrees=degrees,
            working_dps=working_dps,
        )


def assemble_settling_legendre_system(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    settling_velocity_m_s: float,
    boundary: LowerBoundaryFluxLaw,
    n_quad: int = 256,
    z_lower: float = 0.0,
) -> SettlingLegendreSystem:
    """Assemble the conservative steady settling operator."""
    vg = _nonnegative("settling_velocity_m_s", settling_velocity_m_s)
    if float(boundary.equilibrium_concentration) != 0.0:
        raise ValueError("QA-038 settling transport accepts only zero-equilibrium aerosol boundary laws")

    h, z_lower, n_modes, M, S, rhs, y0 = _assemble_interior(
        h=h, n_modes=n_modes, wind=wind, diffusivity=diffusivity,
        source_height=source_height, emission_rate=emission_rate,
        n_quad=n_quad, z_lower=z_lower,
    )
    zq, wq = gauss_legendre_interval(z_lower, h, int(n_quad))
    phi = legendre_values(zq, h=h, n_modes=n_modes, z_lower=z_lower)
    dphi = legendre_derivatives(zq, h=h, n_modes=n_modes, z_lower=z_lower)
    W = dphi.T @ ((wq*vg)[:, None] * phi)

    b = lower_values(h=h, n_modes=n_modes, z_lower=z_lower)
    terms = coerce_boundary_weak_terms(boundary, b)
    if np.any(terms.forcing != 0.0):
        raise ValueError("QA-038 aerosol boundary must be homogeneous")
    A = S + W + terms.matrix

    # Exact integration-by-parts identity for constant Vg.  This detects the
    # most dangerous sign/transposition errors in the drift operator.
    t = top_values(h=h, n_modes=n_modes, z_lower=z_lower)
    expected_sym = vg * (np.outer(t, t) - np.outer(b, b))
    # The identity is exact analytically, but high-order shifted-Legendre
    # derivative evaluation and Gauss-Legendre node/weight construction amplify
    # floating-point roundoff approximately with polynomial order.  Use a
    # scale-aware O(eps*N^2) verification tolerance rather than a fixed absolute
    # threshold.  This remains many orders of magnitude below a material
    # sign/transposition error and changes only the internal QA guard, not W.
    observed_sym = W + W.T
    scale = max(1.0, float(np.max(np.abs(expected_sym))), float(np.max(np.abs(observed_sym))))
    ibp_tol = 4096.0 * np.finfo(float).eps * (float(n_modes) ** 2) * scale
    if float(np.max(np.abs(observed_sym - expected_sym))) > ibp_tol:
        raise FloatingPointError("settling operator failed integration-by-parts identity")

    return SettlingLegendreSystem(
        z_lower=z_lower, h=h, n_modes=n_modes,
        settling_velocity_m_s=vg,
        boundary_sink_velocity_m_s=float(boundary.sink_velocity_m_s),
        boundary_label=str(boundary.label),
        M=M, diffusion_operator=S, settling_operator=np.asarray(W, dtype=np.float64),
        boundary_matrix=terms.matrix, operator=np.asarray(A, dtype=np.float64),
        rhs_source=rhs, y0=y0,
    )


def assemble_transient_settling_legendre_system(**kwargs) -> TransientSettlingLegendreSystem:
    """Assemble the transient/Laplace settling system from the steady operator."""
    steady = assemble_settling_legendre_system(**kwargs)
    G = np.eye(steady.n_modes, dtype=np.float64)
    return TransientSettlingLegendreSystem(steady=steady, G=G)
