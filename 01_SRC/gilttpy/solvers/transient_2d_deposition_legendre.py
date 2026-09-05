"""Modern shifted-Legendre transient solver with typed affine boundary flux.

For a time-independent lower-interface law switched on at t=0,

    J_down(t) = k [C_lower(t) - C_eq],

the temporal Laplace transform gives the x-domain system

    M Y_x + (A + s G)Y = f/s,
    M Y(0,s) = (Q/s) phi(Hs),

with ``A=S+kbb^T`` and ``f=k*C_eq*b``.  The ``1/s`` on the boundary forcing
is essential: it is the transform of the constant compensation state.  When
``C_eq=0`` the implementation follows the pre-QA037 homogeneous path exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, solve

from gilttpy.basis.shifted_legendre import values as legendre_values
from gilttpy.solvers.lower_boundary_operator import (
    LinearRobinBoundaryCondition,
    LowerBoundaryFluxLaw,
)
from gilttpy.solvers.steady_2d_deposition_legendre import (
    LegendreDepositionSystem,
    assemble_legendre_boundary_system,
    assemble_legendre_deposition_system,
)
from gilttpy.solvers.transient_2d import fixed_talbot_inverse
from gilttpy.numerics.inverse_laplace_modern import (
    dehoog_inverse_laplace,
    dehoog_consensus_inverse_laplace,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Profile = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class TransientLegendreDepositionSystem:
    z_lower: float
    h: float
    n_modes: int
    deposition_velocity: float
    equilibrium_concentration: float
    boundary_label: str
    M: FloatArray
    operator: FloatArray
    boundary_forcing: FloatArray
    G: FloatArray
    source_coefficients: FloatArray

    @property
    def interval_length(self) -> float:
        return float(self.h-self.z_lower)

    @property
    def boundary_is_affine(self) -> bool:
        return bool(np.any(self.boundary_forcing != 0.0))

    def _validate_s(self, s: complex) -> complex:
        s=complex(s)
        if not np.isfinite(s.real) or not np.isfinite(s.imag) or abs(s)==0.0:
            raise ValueError("s must be finite and nonzero")
        return s

    def laplace_system_matrix(self, s: complex) -> ComplexArray:
        s=self._validate_s(s)
        return np.asarray(
            self.operator.astype(np.complex128,copy=False)
            + s*self.G.astype(np.complex128,copy=False),
            dtype=np.complex128,
        )

    def laplace_generator(self, s: complex) -> ComplexArray:
        s=self._validate_s(s)
        return -solve(
            self.M.astype(np.complex128,copy=False),
            self.laplace_system_matrix(s),
            assume_a="gen",check_finite=True,
        )

    def laplace_x_forcing(self, s: complex) -> ComplexArray:
        """Return ``M^-1 (f/s)`` in the x-domain ODE."""
        s=self._validate_s(s)
        if not self.boundary_is_affine:
            return np.zeros(self.n_modes,dtype=np.complex128)
        return solve(
            self.M.astype(np.complex128,copy=False),
            self.boundary_forcing.astype(np.complex128,copy=False)/s,
            assume_a="gen",check_finite=True,
        )

    def laplace_coefficients(self, x: float, s: complex) -> ComplexArray:
        x=float(x); s=self._validate_s(s)
        if not np.isfinite(x) or x<0.0:
            raise ValueError("x must be finite and non-negative")
        y0=self.source_coefficients.astype(np.complex128,copy=False)/s
        generator=self.laplace_generator(s)
        if not self.boundary_is_affine:
            return expm(generator*x)@y0
        K=self.laplace_system_matrix(s)
        y_eq=solve(
            K,
            self.boundary_forcing.astype(np.complex128,copy=False)/s,
            assume_a="gen",check_finite=True,
        )
        return y_eq + expm(generator*x)@(y0-y_eq)

    def laplace_state(self, x: float, s: complex) -> tuple[ComplexArray,ComplexArray]:
        """Return ``(Y,dY/dx)`` for the affine Laplace-domain system."""
        s=self._validate_s(s)
        y=self.laplace_coefficients(x,s)
        yp=self.laplace_generator(s)@y + self.laplace_x_forcing(s)
        return y,yp

    def laplace_coefficients_derivative(self, x: float, s: complex) -> ComplexArray:
        return self.laplace_state(x,s)[1]

    def laplace_residual(self, x: float, s: complex) -> ComplexArray:
        """Return ``M Y_x + (A+sG)Y - f/s``."""
        s=self._validate_s(s)
        y,yp=self.laplace_state(x,s)
        return (
            self.M.astype(np.complex128,copy=False)@yp
            + self.laplace_system_matrix(s)@y
            - self.boundary_forcing.astype(np.complex128,copy=False)/s
        )

    def laplace_concentration(self, x: float, z: float, s: complex) -> complex:
        z=float(z)
        if not np.isfinite(z) or z<self.z_lower or z>self.h:
            raise ValueError("z outside [z_lower,h]")
        phi=legendre_values(np.asarray([z]),h=self.h,n_modes=self.n_modes,z_lower=self.z_lower)[0]
        return complex(phi@self.laplace_coefficients(x,s))

    def concentration_fixed_talbot(self, x: float, z: float, t: float, *, mstar: int=9) -> float:
        return fixed_talbot_inverse(lambda s:self.laplace_concentration(x,z,s),t,mstar=mstar)

    def concentration_dehoog(self, x: float, z: float, t: float, *, degree: int=28, working_dps: int=40) -> float:
        """Modern single-degree de Hoog path; Fixed Talbot remains a historical comparator."""
        return dehoog_inverse_laplace(
            lambda s:self.laplace_concentration(x,z,s),
            t,degree=degree,working_dps=working_dps,
        )

    def concentration_dehoog_consensus(
        self,
        x: float,
        z: float,
        t: float,
        *,
        degrees: tuple[int, ...]=(24,26,28),
        working_dps: int=40,
    ) -> float:
        """Portable degree-consensus path for complex128 spectral evaluations."""
        return dehoog_consensus_inverse_laplace(
            lambda s:self.laplace_concentration(x,z,s),
            t,degrees=degrees,working_dps=working_dps,
        )

    def lower_boundary_concentration_fixed_talbot(self, x: float, t: float, *, mstar: int=9) -> float:
        return self.concentration_fixed_talbot(x,self.z_lower,t,mstar=mstar)


def _from_steady(steady: LegendreDepositionSystem) -> TransientLegendreDepositionSystem:
    G=np.eye(steady.n_modes,dtype=np.float64)
    return TransientLegendreDepositionSystem(
        z_lower=steady.z_lower,h=steady.h,n_modes=steady.n_modes,
        deposition_velocity=steady.deposition_velocity,
        equilibrium_concentration=steady.equilibrium_concentration,
        boundary_label=steady.boundary_label,
        M=steady.M,operator=steady.operator,boundary_forcing=steady.boundary_forcing,
        G=G,source_coefficients=steady.y0,
    )


def assemble_transient_legendre_boundary_system(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    boundary: LowerBoundaryFluxLaw,
    n_quad: int=256,
    z_lower: float=0.0,
) -> TransientLegendreDepositionSystem:
    """Assemble the transient system from the verified typed steady operator."""
    steady=assemble_legendre_boundary_system(
        h=h,n_modes=n_modes,wind=wind,diffusivity=diffusivity,
        source_height=source_height,emission_rate=emission_rate,
        boundary=boundary,n_quad=n_quad,z_lower=z_lower,
    )
    return _from_steady(steady)


def assemble_transient_legendre_deposition_system(
    *,
    h: float,
    n_modes: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    deposition_velocity: float,
    n_quad: int=256,
    z_lower: float=0.0,
) -> TransientLegendreDepositionSystem:
    """Backward-compatible homogeneous Robin wrapper retained for QA-027 lineage."""
    steady=assemble_legendre_deposition_system(
        h=h,n_modes=n_modes,wind=wind,diffusivity=diffusivity,
        source_height=source_height,emission_rate=emission_rate,
        deposition_velocity=deposition_velocity,n_quad=n_quad,z_lower=z_lower,
    )
    return _from_steady(steady)
