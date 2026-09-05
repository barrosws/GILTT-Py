"""QA-043 transient variable-coefficient verification references.

This module provides two independent target-free references for the modern
transient transport chain:

1. An exact causal eigenmode with constant positive wind, variable K(z),
   non-zero gravitational settling and a homogeneous total-flux Robin lower
   boundary.  Its time-domain solution contains an exact advective delay and
   therefore directly tests causality and inverse-Laplace robustness.
2. A direct time-domain method-of-lines reference using P1 finite elements in
   z and first-order upwind finite volumes in x.  It is intentionally distinct
   from shifted-Legendre + Laplace inversion and is used only for validation.

No observational target, historical GILTT output, clipping, or target tuning is
used in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, solve
from scipy.sparse import bmat, csr_matrix, hstack, vstack
from scipy.sparse.linalg import expm_multiply

from gilttpy.basis.shifted_legendre import values as legendre_values
from gilttpy.numerics.inverse_laplace_modern import dehoog_inverse_laplace

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
class ExactTransientVariableDiffusivityCase:
    """Exact delayed transient with variable diffusivity and settling.

    Let xi=z-z_lower, L=h-z_lower and phi=A exp(-a xi).  Choose

      K(z) = Vg/a + mu/a^2 [exp(a(xi-L))-1],
      k_lower = mu/a [1-exp(-aL)].

    Then d/dz(K phi_z + Vg phi) = -mu phi, the top total flux is zero,
    and the lower total flux equals k_lower*phi.  For constant wind u0 and a
    unit step inlet phi(z) H(t), the exact solution is

      C(x,z,t)=phi(z) exp(-mu*x/u0) H(t-x/u0).

    The amplitude is normalized so integral u0*phi dz = 1.
    """

    z_lower: float = 0.0
    h: float = 10.0
    wind_m_s: float = 2.0
    vertical_decay_per_m: float = 0.15
    eigen_decay_per_s: float = 0.03
    settling_velocity_m_s: float = 0.35
    label: str = "QA043 exact causal variable-diffusivity eigenmode"
    provenance: str = "QA043 analytical construction; no observational target"

    def __post_init__(self) -> None:
        zl = _finite("z_lower", self.z_lower)
        h = _finite("h", self.h)
        if h <= zl:
            raise ValueError("require h > z_lower")
        u = _positive("wind_m_s", self.wind_m_s)
        a = _positive("vertical_decay_per_m", self.vertical_decay_per_m)
        mu = _positive("eigen_decay_per_s", self.eigen_decay_per_s)
        vg = _positive("settling_velocity_m_s", self.settling_velocity_m_s)
        # K(z_lower)>0 is the exact applicability condition.
        if vg/a - mu/a**2*(1.0-math.exp(-a*(h-zl))) <= 0.0:
            raise ValueError("chosen parameters make lower diffusivity nonpositive")
        if not str(self.label).strip() or not str(self.provenance).strip():
            raise ValueError("label and provenance are required")
        del u

    @property
    def length(self) -> float:
        return float(self.h-self.z_lower)

    @property
    def a(self) -> float:
        return float(self.vertical_decay_per_m)

    @property
    def mu(self) -> float:
        return float(self.eigen_decay_per_s)

    @property
    def inlet_amplitude(self) -> float:
        return float(self.a/(self.wind_m_s*(1.0-math.exp(-self.a*self.length))))

    @property
    def boundary_sink_velocity_m_s(self) -> float:
        return float(self.mu/self.a*(1.0-math.exp(-self.a*self.length)))

    def wind(self, z: ArrayLike) -> FloatArray:
        zz = np.asarray(z, dtype=np.float64)
        return np.full_like(zz, self.wind_m_s, dtype=np.float64)

    def diffusivity(self, z: ArrayLike) -> FloatArray:
        zz = np.asarray(z, dtype=np.float64)
        xi = zz-self.z_lower
        if np.any(xi < 0.0) or np.any(xi > self.length) or not np.all(np.isfinite(zz)):
            raise ValueError("z outside exact-case interval")
        out = (
            self.settling_velocity_m_s/self.a
            + self.mu/self.a**2*(np.exp(self.a*(xi-self.length))-1.0)
        )
        return np.asarray(out, dtype=np.float64)

    def inlet_profile(self, z: ArrayLike) -> FloatArray:
        zz = np.asarray(z, dtype=np.float64)
        xi = zz-self.z_lower
        if np.any(xi < 0.0) or np.any(xi > self.length) or not np.all(np.isfinite(zz)):
            raise ValueError("z outside exact-case interval")
        return np.asarray(self.inlet_amplitude*np.exp(-self.a*xi), dtype=np.float64)

    def downward_vertical_flux(self, z: ArrayLike) -> FloatArray:
        c = self.inlet_profile(z)
        return np.asarray(
            (-self.a*self.diffusivity(z)+self.settling_velocity_m_s)*c,
            dtype=np.float64,
        )

    def vertical_flux_derivative(self, z: ArrayLike) -> FloatArray:
        return np.asarray(-self.mu*self.inlet_profile(z), dtype=np.float64)

    def arrival_time_s(self, x: float) -> float:
        x = _finite("x", x)
        if x < 0.0:
            raise ValueError("x must be nonnegative")
        return float(x/self.wind_m_s)

    def exact_concentration(self, x: float, z: ArrayLike, t: float) -> FloatArray:
        x = _finite("x", x)
        t = _finite("t", t)
        if x < 0.0 or t < 0.0:
            raise ValueError("x and t must be nonnegative")
        amp = math.exp(-self.mu*x/self.wind_m_s) if t >= self.arrival_time_s(x) else 0.0
        return np.asarray(amp*self.inlet_profile(z), dtype=np.float64)

    def exact_laplace_concentration(self, x: float, z: ArrayLike, s: complex) -> NDArray[np.complex128]:
        x = _finite("x", x)
        s = complex(s)
        if x < 0.0 or not np.isfinite(s.real) or not np.isfinite(s.imag) or abs(s) == 0.0:
            raise ValueError("invalid x or s")
        return np.asarray(
            self.inlet_profile(z).astype(np.complex128)
            * np.exp(-(s+self.mu)*x/self.wind_m_s)/s,
            dtype=np.complex128,
        )


def spectral_laplace_coefficients_from_inlet(system, inlet_coefficients: ArrayLike, x: float, s: complex):
    """Laplace coefficients for an explicit validation inlet vector."""
    y0 = np.asarray(inlet_coefficients, dtype=np.float64)
    if y0.shape != (system.n_modes,) or not np.all(np.isfinite(y0)):
        raise ValueError("invalid inlet coefficient vector")
    x = _finite("x", x)
    s = complex(s)
    if x < 0.0 or not np.isfinite(s.real) or not np.isfinite(s.imag) or abs(s) == 0.0:
        raise ValueError("invalid x or s")
    matrix = system.operator.astype(np.complex128) + s*np.eye(system.n_modes, dtype=np.complex128)
    generator = -solve(system.M.astype(np.complex128), matrix, assume_a="gen", check_finite=True)
    return expm(generator*x) @ (y0.astype(np.complex128)/s)


def spectral_laplace_concentration_from_inlet(
    system, inlet_coefficients: ArrayLike, x: float, z: float, s: complex,
) -> complex:
    z = _finite("z", z)
    if z < system.z_lower or z > system.h:
        raise ValueError("z outside system interval")
    phi = legendre_values(
        np.asarray([z]), h=system.h, n_modes=system.n_modes, z_lower=system.z_lower
    )[0]
    return complex(phi @ spectral_laplace_coefficients_from_inlet(system, inlet_coefficients, x, s))


def spectral_dehoog_concentration_from_inlet(
    system,
    inlet_coefficients: ArrayLike,
    x: float,
    z: float,
    t: float,
    *,
    degree: int = 28,
    working_dps: int = 40,
) -> float:
    return dehoog_inverse_laplace(
        lambda s: spectral_laplace_concentration_from_inlet(system, inlet_coefficients, x, z, s),
        t,
        degree=degree,
        working_dps=working_dps,
    )


@dataclass(frozen=True)
class DirectTransientFEMFVSystem:
    """Validation-only P1-FEM(z) + upwind-FV(x) direct time reference."""

    z_lower: float
    h: float
    x_end: float
    n_x: int
    nodes_z: FloatArray
    dx: float
    augmented_generator: object

    @property
    def n_z(self) -> int:
        return int(self.nodes_z.size)

    def state(self, t: float) -> FloatArray:
        t = _finite("t", t)
        if t < 0.0:
            raise ValueError("t must be nonnegative")
        n = self.n_x*self.n_z
        initial = np.zeros(n+1, dtype=np.float64)
        initial[-1] = 1.0
        out = expm_multiply(self.augmented_generator*t, initial)[:-1]
        return np.asarray(out.reshape(self.n_x, self.n_z), dtype=np.float64)

    def concentration(self, x: float, z: ArrayLike, t: float) -> FloatArray:
        x = _positive("x", x)
        # The first FV state is centered/represented at x=dx under the backward
        # upwind derivative.  Validation queries are required to lie on that grid.
        j_float = x/self.dx
        j = int(round(j_float))
        if j < 1 or j > self.n_x or abs(j_float-j) > 1e-10:
            raise ValueError("x must coincide with a positive direct-reference grid point")
        zz = np.asarray(z, dtype=np.float64)
        if np.any(zz < self.z_lower) or np.any(zz > self.h) or not np.all(np.isfinite(zz)):
            raise ValueError("z outside direct-reference interval")
        row = self.state(t)[j-1]
        return np.asarray(np.interp(zz, self.nodes_z, row), dtype=np.float64)


def assemble_direct_transient_fem_fv_reference(
    *,
    z_lower: float,
    h: float,
    x_end: float,
    n_x: int,
    n_z_elements: int,
    wind: Profile,
    diffusivity: Profile,
    settling_velocity_m_s: float,
    boundary_sink_velocity_m_s: float,
    inlet_profile: Profile,
    element_quad_order: int = 5,
) -> DirectTransientFEMFVSystem:
    """Assemble independent direct-time validation discretization.

    Weak z discretization gives G y_t + M_u y_x + A y = 0.  Positive wind is
    discretized by backward/upwind FV in x.  The resulting linear MOL system is
    advanced exactly in time for the semi-discrete equations with sparse
    ``expm_multiply`` and a constant step boundary state at x=0.
    """
    zl = _finite("z_lower", z_lower)
    h = _finite("h", h)
    xe = _positive("x_end", x_end)
    if h <= zl:
        raise ValueError("require h > z_lower")
    if int(n_x) != n_x or int(n_x) < 4:
        raise ValueError("n_x must be integer >=4")
    if int(n_z_elements) != n_z_elements or int(n_z_elements) < 4:
        raise ValueError("n_z_elements must be integer >=4")
    vg = _positive("settling_velocity_m_s", settling_velocity_m_s)
    kb = _positive("boundary_sink_velocity_m_s", boundary_sink_velocity_m_s)
    if int(element_quad_order) != element_quad_order or int(element_quad_order) < 2:
        raise ValueError("element_quad_order must be integer >=2")

    n_x = int(n_x)
    ne = int(n_z_elements)
    nodes = np.linspace(zl, h, ne+1, dtype=np.float64)
    nz = nodes.size
    G = np.zeros((nz, nz), dtype=np.float64)
    Mu = np.zeros((nz, nz), dtype=np.float64)
    A = np.zeros((nz, nz), dtype=np.float64)
    gx, gw = leggauss(int(element_quad_order))

    for e in range(ne):
        za = float(nodes[e]); zb = float(nodes[e+1]); dz = zb-za
        zq = 0.5*(za+zb)+0.5*dz*gx
        wq = 0.5*dz*gw
        N = np.column_stack(((zb-zq)/dz, (zq-za)/dz))
        dN = np.array([-1.0/dz, 1.0/dz], dtype=np.float64)
        u = np.asarray(wind(zq), dtype=np.float64)
        k = np.asarray(diffusivity(zq), dtype=np.float64)
        if u.shape != zq.shape or not np.all(np.isfinite(u)) or np.any(u <= 0.0):
            raise ValueError("wind must be finite positive on direct-reference elements")
        if k.shape != zq.shape or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
            raise ValueError("diffusivity must be finite positive on direct-reference elements")
        Ge = N.T @ (wq[:, None]*N)
        Me = N.T @ ((wq*u)[:, None]*N)
        Se = np.outer(dN, dN)*float(np.sum(wq*k))
        int_N = np.sum(wq[:, None]*N, axis=0)
        We = vg*np.outer(dN, int_N)
        ix = np.ix_([e, e+1], [e, e+1])
        G[ix] += Ge
        Mu[ix] += Me
        A[ix] += Se+We

    A[0, 0] += kb
    evals = np.linalg.eigvalsh(0.5*(G+G.T))
    if evals[0] <= 100*np.finfo(float).eps*max(1.0, evals[-1]):
        raise FloatingPointError("direct-reference time mass matrix is not positive definite")

    Ginv_A = solve(G, A, assume_a="pos", check_finite=True)
    Ginv_M = solve(G, Mu, assume_a="pos", check_finite=True)
    dx = xe/n_x
    H0 = -(Ginv_A+Ginv_M/dx)
    Hp = Ginv_M/dx

    H0s = csr_matrix(H0)
    Hps = csr_matrix(Hp)
    blocks = [[None]*n_x for _ in range(n_x)]
    for j in range(n_x):
        blocks[j][j] = H0s
        if j > 0:
            blocks[j][j-1] = Hps
    H = bmat(blocks, format="csr")

    yin = np.asarray(inlet_profile(nodes), dtype=np.float64)
    if yin.shape != nodes.shape or not np.all(np.isfinite(yin)):
        raise ValueError("inlet_profile must return finite nodal values")
    forcing = np.zeros(n_x*nz, dtype=np.float64)
    forcing[:nz] = Hp @ yin
    augmented = vstack(
        [
            hstack([H, csr_matrix(forcing[:, None])]),
            csr_matrix((1, n_x*nz+1)),
        ],
        format="csr",
    )
    return DirectTransientFEMFVSystem(zl, h, xe, n_x, nodes, dx, augmented)
