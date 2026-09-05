"""Independent variable-coefficient benchmark for QA-041.

This module is validation-only.  It constructs an exact exponentially decaying
solution of the same conservative steady aerosol transport equation used by
QA-038,

    u(z) C_x = d/dz [K(z) C_z + Vg C],

on an explicit interval [z_lower, h].  The coefficient family is manufactured
so that wind and diffusivity both vary with height, settling is non-zero, the
top total flux is exactly zero, and the lower total flux is an exact homogeneous
Robin law.  With the normalization used here the initial advective throughflow
is exactly one source unit, so

    F_adv(x) = exp(-lambda_x x),
    J_lower(x) = lambda_x exp(-lambda_x x),
    D(0,x) = 1-exp(-lambda_x x).

The module also contains a continuous piecewise-linear FEM reference.  The FEM
uses nodal basis functions and element quadrature and is deliberately distinct
from the shifted-Legendre spectral discretization.  It is a verification
reference, not a production solver and not a historical GILTT branch.

The design follows the general code-verification principle of comparing a
numerical implementation against a known exact/manufactured solution and then
checking convergence under an independent discretization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, solve

from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.basis.shifted_legendre import values as legendre_values

FloatArray = NDArray[np.float64]
Profile = Callable[[FloatArray], ArrayLike]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ManufacturedVariableCoefficientCase:
    """Exact variable-coefficient conservative settling eigenmode.

    Let xi=z-z_lower, L=h-z_lower and

        K(z) = K0 [1 + beta xi/L],
        C(x,z) = A exp(-a xi) exp(-lambda_x x),
        Vg = a K(h),
        k_lower = a [K(h)-K(z_lower)] = a K0 beta.

    Choosing

        u(z) = a K0 beta [1/L + a(1-xi/L)] / lambda_x

    makes the conservative PDE hold exactly.  With A=lambda_x/k_lower,
    the x=0 advective flux is exactly one.
    """

    z_lower: float = 10.0
    h: float = 110.0
    diffusivity_lower_m2_s: float = 1.5
    diffusivity_fractional_increase: float = 0.8
    vertical_decay_per_m: float = 0.02
    streamwise_decay_per_m: float = 2.0e-4
    label: str = "QA041 manufactured variable-coefficient eigenmode"
    provenance: str = "QA041 exact algebraic construction; no observational target"

    def __post_init__(self) -> None:
        zl = _finite("z_lower", self.z_lower)
        h = _finite("h", self.h)
        if h <= zl:
            raise ValueError("require h > z_lower")
        _positive("diffusivity_lower_m2_s", self.diffusivity_lower_m2_s)
        _positive("diffusivity_fractional_increase", self.diffusivity_fractional_increase)
        _positive("vertical_decay_per_m", self.vertical_decay_per_m)
        _positive("streamwise_decay_per_m", self.streamwise_decay_per_m)
        if not str(self.label).strip() or not str(self.provenance).strip():
            raise ValueError("label and provenance are required")

    @property
    def length(self) -> float:
        return float(self.h-self.z_lower)

    @property
    def k0(self) -> float:
        return float(self.diffusivity_lower_m2_s)

    @property
    def beta(self) -> float:
        return float(self.diffusivity_fractional_increase)

    @property
    def a(self) -> float:
        return float(self.vertical_decay_per_m)

    @property
    def lambda_x(self) -> float:
        return float(self.streamwise_decay_per_m)

    @property
    def settling_velocity_m_s(self) -> float:
        return self.a*self.k0*(1.0+self.beta)

    @property
    def boundary_sink_velocity_m_s(self) -> float:
        return self.a*self.k0*self.beta

    @property
    def amplitude(self) -> float:
        return self.lambda_x/self.boundary_sink_velocity_m_s

    def _z(self, z: ArrayLike) -> FloatArray:
        zz = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if (not np.all(np.isfinite(zz)) or np.any(zz < self.z_lower)
                or np.any(zz > self.h)):
            raise ValueError("z outside manufactured interval")
        return zz

    def diffusivity(self, z: ArrayLike) -> FloatArray:
        zz = self._z(z)
        xi = zz-self.z_lower
        return np.asarray(self.k0*(1.0+self.beta*xi/self.length), dtype=np.float64)

    def wind(self, z: ArrayLike) -> FloatArray:
        zz = self._z(z)
        xi = zz-self.z_lower
        bracket = 1.0/self.length + self.a*(1.0-xi/self.length)
        return np.asarray(
            self.a*self.k0*self.beta*bracket/self.lambda_x,
            dtype=np.float64,
        )

    def concentration(self, x: float, z: ArrayLike) -> FloatArray:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        zz = self._z(z)
        return np.asarray(
            self.amplitude*np.exp(-self.a*(zz-self.z_lower)-self.lambda_x*x),
            dtype=np.float64,
        )

    def concentration_gradient(self, x: float, z: ArrayLike) -> FloatArray:
        return -self.a*self.concentration(x, z)

    def streamwise_derivative(self, x: float, z: ArrayLike) -> FloatArray:
        return -self.lambda_x*self.concentration(x, z)

    def downward_vertical_flux(self, x: float, z: ArrayLike) -> FloatArray:
        c = self.concentration(x, z)
        return self.diffusivity(z)*(-self.a*c) + self.settling_velocity_m_s*c

    def vertical_flux_derivative(self, x: float, z: ArrayLike) -> FloatArray:
        zz = self._z(z)
        xi = zz-self.z_lower
        c = self.concentration(x, zz)
        return np.asarray(
            -self.a*self.k0*self.beta*(1.0/self.length + self.a*(1.0-xi/self.length))*c,
            dtype=np.float64,
        )

    def exact_advective_flux(self, x: float) -> float:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        return float(math.exp(-self.lambda_x*x))

    def exact_lower_flux(self, x: float) -> float:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        return float(self.lambda_x*math.exp(-self.lambda_x*x))

    def exact_integrated_deposition(self, x: float) -> float:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        return float(1.0-math.exp(-self.lambda_x*x))


def project_profile_to_legendre(
    profile: Callable[[FloatArray], ArrayLike],
    *, h: float, n_modes: int, z_lower: float, n_quad: int = 512,
) -> FloatArray:
    """L2-project a smooth validation profile onto the orthonormal basis."""
    zq, wq = gauss_legendre_interval(float(z_lower), float(h), int(n_quad))
    phi = legendre_values(zq, h=float(h), n_modes=int(n_modes), z_lower=float(z_lower))
    val = np.asarray(profile(zq), dtype=np.float64)
    if val.shape != zq.shape or not np.all(np.isfinite(val)):
        raise ValueError("profile must return finite values with matching shape")
    return np.asarray(phi.T @ (wq*val), dtype=np.float64)


def spectral_solution_from_initial_coefficients(system, y0: ArrayLike, x: float, z: ArrayLike) -> FloatArray:
    """Propagate supplied smooth initial coefficients through a frozen spectral operator."""
    yy = np.asarray(y0, dtype=np.float64)
    if yy.shape != (system.n_modes,) or not np.all(np.isfinite(yy)):
        raise ValueError("y0 shape mismatch or nonfinite values")
    x = _finite("x", x)
    if x < 0.0:
        raise ValueError("x must be nonnegative")
    coeff = expm(system.generator*x) @ yy
    return np.asarray(
        legendre_values(np.asarray(z, dtype=np.float64), h=system.h,
                        n_modes=system.n_modes, z_lower=system.z_lower) @ coeff,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class FEMVariableCoefficientSystem:
    """Independent P1-FEM reference for the QA-041 conservative operator."""

    z_lower: float
    h: float
    nodes: FloatArray
    settling_velocity_m_s: float
    boundary_sink_velocity_m_s: float
    M: FloatArray
    operator: FloatArray
    y0: FloatArray

    @property
    def n_elements(self) -> int:
        return int(self.nodes.size-1)

    @property
    def generator(self) -> FloatArray:
        return np.asarray(-solve(self.M, self.operator, assume_a="gen", check_finite=True), dtype=np.float64)

    def coefficients(self, x: float) -> FloatArray:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        return np.asarray(np.real_if_close(expm(self.generator*x) @ self.y0, tol=1000), dtype=np.float64)

    def concentration(self, x: float, z: ArrayLike) -> FloatArray:
        zz = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if (not np.all(np.isfinite(zz)) or np.any(zz < self.z_lower)
                or np.any(zz > self.h)):
            raise ValueError("z outside FEM interval")
        y = self.coefficients(x)
        return np.asarray(np.interp(zz, self.nodes, y), dtype=np.float64)

    def lower_boundary_concentration(self, x: float) -> float:
        return float(self.coefficients(x)[0])

    def lower_boundary_downward_flux(self, x: float) -> float:
        return self.boundary_sink_velocity_m_s*self.lower_boundary_concentration(x)

    def flux_derivative_from_constant_test(self, x: float) -> float:
        y = self.coefficients(x)
        yp = self.generator @ y
        one = np.ones(self.nodes.size, dtype=np.float64)
        return float(one @ (self.M @ yp))


def assemble_fem_variable_coefficient_reference(
    *,
    z_lower: float,
    h: float,
    n_elements: int,
    wind: Profile,
    diffusivity: Profile,
    settling_velocity_m_s: float,
    boundary_sink_velocity_m_s: float,
    initial_profile: Callable[[FloatArray], ArrayLike],
    element_quad_order: int = 6,
) -> FEMVariableCoefficientSystem:
    """Assemble an independent uniform-mesh P1 FEM validation reference."""
    zl = _finite("z_lower", z_lower); h = _finite("h", h)
    if h <= zl:
        raise ValueError("require h > z_lower")
    if int(n_elements) != n_elements or int(n_elements) < 4:
        raise ValueError("n_elements must be an integer >=4")
    vg = _positive("settling_velocity_m_s", settling_velocity_m_s)
    kb = _positive("boundary_sink_velocity_m_s", boundary_sink_velocity_m_s)
    if int(element_quad_order) != element_quad_order or int(element_quad_order) < 2:
        raise ValueError("element_quad_order must be integer >=2")

    nodes = np.linspace(zl, h, int(n_elements)+1, dtype=np.float64)
    n = nodes.size
    M = np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    gx, gw = leggauss(int(element_quad_order))

    for e in range(n-1):
        za = float(nodes[e]); zb = float(nodes[e+1]); dz = zb-za
        zq = 0.5*(za+zb)+0.5*dz*gx
        wq = 0.5*dz*gw
        N = np.column_stack(((zb-zq)/dz, (zq-za)/dz))
        dN = np.array([-1.0/dz, 1.0/dz], dtype=np.float64)
        u = np.asarray(wind(zq), dtype=np.float64)
        k = np.asarray(diffusivity(zq), dtype=np.float64)
        if u.shape != zq.shape or k.shape != zq.shape:
            raise ValueError("profile shape mismatch")
        if not np.all(np.isfinite(u)) or np.any(u <= 0.0):
            raise ValueError("wind must be finite and positive")
        if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
            raise ValueError("diffusivity must be finite and positive")

        Me = N.T @ ((wq*u)[:, None]*N)
        Se = np.outer(dN, dN)*float(np.sum(wq*k))
        int_N = np.sum(wq[:, None]*N, axis=0)
        We = vg*np.outer(dN, int_N)
        ix = np.ix_([e, e+1], [e, e+1])
        M[ix] += Me
        A[ix] += Se+We

    A[0, 0] += kb
    M = 0.5*(M+M.T)
    evals = np.linalg.eigvalsh(M)
    if evals[0] <= 100*np.finfo(float).eps*max(1.0, evals[-1]):
        raise FloatingPointError("FEM wind mass matrix is not numerically positive definite")

    y0 = np.asarray(initial_profile(nodes), dtype=np.float64)
    if y0.shape != nodes.shape or not np.all(np.isfinite(y0)):
        raise ValueError("initial_profile must return finite nodal values")

    return FEMVariableCoefficientSystem(
        z_lower=zl,
        h=h,
        nodes=nodes,
        settling_velocity_m_s=vg,
        boundary_sink_velocity_m_s=kb,
        M=M,
        operator=np.asarray(A, dtype=np.float64),
        y0=y0,
    )
