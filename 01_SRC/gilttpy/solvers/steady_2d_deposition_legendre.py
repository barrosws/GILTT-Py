"""Modern shifted-Legendre Galerkin solver with typed lower-boundary flux.

QA-037 generalizes the QA-025/026 homogeneous deposition sink to the affine
resolved-interface law frozen by QA-036,

    J_down = k * (C(z_lower) - C_eq).

The conservative finite-dimensional problem is

    M Y'(x) + A Y(x) = f,
    A = S + B,
    B = k b b^T,
    f = k C_eq b,

where ``b`` evaluates the basis at ``z_lower``.  The historical and previous
modern homogeneous API is preserved by ``assemble_legendre_deposition_system``;
it is now a backward-compatible wrapper with ``C_eq=0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, expm, solve

from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.basis.shifted_legendre import values as legendre_values
from gilttpy.basis.shifted_legendre import derivatives as legendre_derivatives
from gilttpy.basis.shifted_legendre import lower_values
from gilttpy.solvers.lower_boundary_operator import (
    LinearRobinBoundaryCondition,
    LowerBoundaryFluxLaw,
    coerce_boundary_weak_terms,
)

FloatArray = NDArray[np.float64]
Profile = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class LegendreDepositionSystem:
    z_lower: float
    h: float
    n_modes: int
    deposition_velocity: float
    equilibrium_concentration: float
    boundary_label: str
    M: FloatArray
    diffusion_operator: FloatArray
    boundary_matrix: FloatArray
    boundary_forcing: FloatArray
    operator: FloatArray
    rhs_source: FloatArray
    y0: FloatArray
    decay_rates: FloatArray
    eigenvectors: FloatArray

    @property
    def interval_length(self) -> float:
        return float(self.h-self.z_lower)

    @property
    def boundary_is_affine(self) -> bool:
        return bool(np.any(self.boundary_forcing != 0.0))

    def _equilibrium_coefficients(self) -> FloatArray:
        if not self.boundary_is_affine:
            return np.zeros(self.n_modes, dtype=np.float64)
        return np.asarray(
            solve(self.operator, self.boundary_forcing, assume_a="pos", check_finite=True),
            dtype=np.float64,
        )

    def propagate_coefficients(self, initial_coefficients: ArrayLike, x: float, *, method: str = "eig") -> FloatArray:
        """Propagate an arbitrary finite-dimensional inlet state downstream."""
        x = float(x)
        if not np.isfinite(x) or x < 0.0:
            raise ValueError("x must be finite and non-negative")
        y_init = np.asarray(initial_coefficients, dtype=np.float64)
        if y_init.shape != (self.n_modes,) or not np.all(np.isfinite(y_init)):
            raise ValueError("initial_coefficients has invalid shape or values")
        y_eq = self._equilibrium_coefficients()
        delta = y_init - y_eq
        if method == "eig":
            amp = self.eigenvectors.T @ (self.M @ delta)
            y = y_eq + self.eigenvectors @ (np.exp(-self.decay_rates*x)*amp)
        elif method == "expm":
            generator = -solve(self.M, self.operator, assume_a="pos", check_finite=True)
            y = y_eq + expm(generator*x) @ delta
        else:
            raise ValueError("method must be 'eig' or 'expm'")
        return np.asarray(np.real_if_close(y, tol=1000), dtype=np.float64)

    def coefficients_expm(self, x: float) -> FloatArray:
        return self.propagate_coefficients(self.y0, x, method="expm")

    def coefficients_eig(self, x: float) -> FloatArray:
        return self.propagate_coefficients(self.y0, x, method="eig")

    def coefficients(self, x: float) -> FloatArray:
        return self.coefficients_eig(x)

    def coefficients_derivative(self, x: float) -> FloatArray:
        """Exact derivative of the affine finite-dimensional state with respect to x."""
        y = self.coefficients(x)
        return np.asarray(
            -solve(self.M, self.operator @ y - self.boundary_forcing,
                   assume_a="pos", check_finite=True),
            dtype=np.float64,
        )

    def concentration(self, x: float, z: ArrayLike) -> FloatArray:
        z_arr = np.asarray(z, dtype=np.float64)
        return (legendre_values(z_arr, h=self.h, n_modes=self.n_modes,
                                z_lower=self.z_lower) @ self.coefficients(x))

    def lower_boundary_concentration(self, x: float) -> float:
        b = lower_values(h=self.h, n_modes=self.n_modes, z_lower=self.z_lower)
        return float(b @ self.coefficients(x))

    def lower_boundary_downward_flux(self, x: float) -> float:
        return float(self.deposition_velocity) * (
            self.lower_boundary_concentration(x) - float(self.equilibrium_concentration)
        )

    def ground_concentration(self, x: float) -> float:
        """Backward-compatible ground concentration, valid only for z_lower=0."""
        if self.z_lower != 0.0:
            raise ValueError("ground_concentration is undefined when z_lower>0; use lower_boundary_concentration")
        return self.lower_boundary_concentration(x)

    def advective_flux(self, x: float, *, wind: Profile, n_quad: int = 512) -> float:
        zq, wq = gauss_legendre_interval(self.z_lower, self.h, n_quad)
        u = np.asarray(wind(zq), dtype=np.float64)
        return float(np.sum(wq*u*self.concentration(x,zq)))

    def flux_derivative_from_weak_constant_test(self, x: float) -> float:
        """Exact finite-dimensional d/dx integral_[z_lower]^h u C dz."""
        yp = self.coefficients_derivative(x)
        return float(np.sqrt(self.interval_length)*(self.M[0] @ yp))


def _assemble_interior(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    n_quad: int,
    z_lower: float,
) -> tuple[float, float, int, FloatArray, FloatArray, FloatArray, FloatArray]:
    h=float(h); z_lower=float(z_lower); source_height=float(source_height)
    emission_rate=float(emission_rate)
    if not np.isfinite(h) or not np.isfinite(z_lower) or h <= z_lower:
        raise ValueError("require finite h > z_lower")
    if int(n_modes) != n_modes or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    n_modes=int(n_modes)
    if int(n_quad) != n_quad or n_quad < max(2,n_modes):
        raise ValueError("n_quad too small")
    n_quad=int(n_quad)
    if not z_lower <= source_height <= h:
        raise ValueError("source height outside [z_lower,h]")
    if not np.isfinite(emission_rate) or emission_rate < 0.0:
        raise ValueError("invalid emission_rate")

    zq,wq=gauss_legendre_interval(z_lower,h,n_quad)
    u=np.asarray(wind(zq),dtype=np.float64)
    k=np.asarray(diffusivity(zq),dtype=np.float64)
    if u.shape != zq.shape or k.shape != zq.shape:
        raise ValueError("profile shape mismatch")
    if not np.all(np.isfinite(u)) or np.any(u < 0.0) or not np.any(u > 0.0):
        raise ValueError("wind must be finite, nonnegative, and positive on nonzero measure")
    if not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("diffusivity must be finite and nonnegative")

    phi=legendre_values(zq,h=h,n_modes=n_modes,z_lower=z_lower)
    dphi=legendre_derivatives(zq,h=h,n_modes=n_modes,z_lower=z_lower)
    M=phi.T @ ((wq*u)[:,None]*phi); M=0.5*(M+M.T)
    me=np.linalg.eigvalsh(M)
    if me[0] <= 100*np.finfo(float).eps*max(1.0,me[-1]):
        raise FloatingPointError("wind mass matrix is not numerically positive definite")
    S=dphi.T @ ((wq*k)[:,None]*dphi); S=0.5*(S+S.T)
    phs=legendre_values(np.asarray([source_height]),h=h,n_modes=n_modes,z_lower=z_lower)[0]
    rhs=emission_rate*phs
    y0=solve(M,rhs,assume_a="pos",check_finite=True)
    return h, z_lower, n_modes, M, S, rhs, np.asarray(y0,dtype=np.float64)


def assemble_legendre_boundary_system(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    boundary: LowerBoundaryFluxLaw,
    n_quad: int = 256,
    z_lower: float = 0.0,
) -> LegendreDepositionSystem:
    """Assemble the modern steady system with an explicit typed boundary law."""
    h,z_lower,n_modes,M,S,rhs,y0 = _assemble_interior(
        h=h,n_modes=n_modes,wind=wind,diffusivity=diffusivity,
        source_height=source_height,emission_rate=emission_rate,
        n_quad=n_quad,z_lower=z_lower,
    )
    phil=lower_values(h=h,n_modes=n_modes,z_lower=z_lower)
    terms=coerce_boundary_weak_terms(boundary,phil)
    A=S+terms.matrix; A=0.5*(A+A.T)
    mu,vecs=eigh(A,M,check_finite=True)
    scale=max(1.0,float(np.max(np.abs(mu)))); tol=500*np.finfo(float).eps*scale
    if np.min(mu) < -tol:
        raise FloatingPointError("materially negative decay rate")
    mu=np.where(np.abs(mu)<=tol,0.0,mu)
    return LegendreDepositionSystem(
        z_lower=z_lower,h=h,n_modes=n_modes,
        deposition_velocity=float(boundary.sink_velocity_m_s),
        equilibrium_concentration=float(boundary.equilibrium_concentration),
        boundary_label=str(boundary.label),
        M=M,diffusion_operator=S,boundary_matrix=terms.matrix,
        boundary_forcing=terms.forcing,operator=A,rhs_source=rhs,y0=y0,
        decay_rates=np.asarray(mu,dtype=np.float64),
        eigenvectors=np.asarray(vecs,dtype=np.float64),
    )


def assemble_legendre_deposition_system(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    deposition_velocity: float,
    n_quad: int = 256,
    z_lower: float = 0.0,
) -> LegendreDepositionSystem:
    """Backward-compatible homogeneous Robin wrapper retained for prior QA branches."""
    vg=float(deposition_velocity)
    if not np.isfinite(vg) or vg < 0.0:
        raise ValueError("deposition_velocity must be finite and >=0")
    return assemble_legendre_boundary_system(
        h=h,n_modes=n_modes,wind=wind,diffusivity=diffusivity,
        source_height=source_height,emission_rate=emission_rate,
        boundary=LinearRobinBoundaryCondition(
            sink_velocity_m_s=vg,equilibrium_concentration=0.0,
            label="backward-compatible homogeneous deposition boundary",
        ),
        n_quad=n_quad,z_lower=z_lower,
    )
