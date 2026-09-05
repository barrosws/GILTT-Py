"""Independent piecewise-linear finite-element reference solver for steady dry deposition.

This solver is deliberately independent of the spectral GILTT/Legendre bases.  It
solves the same conservative weak problem on an explicit interval [z_lower, h],

    M Y'(x) + (S + Bdep) Y(x) = 0,

with Bdep = Vg e0 e0^T and point-source load M Y(0)=Q N(Hs).
A deterministic power-law graded vertical mesh may be used for verification of
near-ground structure.  It is a QA/reference solver, not a historical branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, solve

FloatArray = NDArray[np.float64]
Profile = Callable[[FloatArray], ArrayLike]


def graded_nodes(*, z_lower: float, h: float, n_elements: int, grading_power: float = 2.0) -> FloatArray:
    z_lower = float(z_lower); h = float(h); grading_power = float(grading_power)
    if not np.isfinite(z_lower) or not np.isfinite(h) or h <= z_lower:
        raise ValueError("require finite h > z_lower")
    if int(n_elements) != n_elements or n_elements < 2:
        raise ValueError("n_elements must be an integer >=2")
    if not np.isfinite(grading_power) or grading_power <= 0.0:
        raise ValueError("grading_power must be finite and >0")
    xi = np.linspace(0.0, 1.0, int(n_elements) + 1)
    nodes = z_lower + (h-z_lower)*xi**grading_power
    if not np.all(np.diff(nodes) > 0.0):
        raise FloatingPointError("mesh nodes are not strictly increasing")
    return np.asarray(nodes, dtype=np.float64)


def _shape_vector(nodes: FloatArray, z: float) -> FloatArray:
    z = float(z)
    if not np.isfinite(z) or z < nodes[0] or z > nodes[-1]:
        raise ValueError("evaluation point outside FEM interval")
    n = nodes.size
    if z == nodes[-1]:
        out = np.zeros(n); out[-1] = 1.0; return out
    e = int(np.searchsorted(nodes, z, side="right") - 1)
    e = max(0, min(e, n-2))
    dz = nodes[e+1]-nodes[e]
    out = np.zeros(n, dtype=np.float64)
    out[e] = (nodes[e+1]-z)/dz
    out[e+1] = (z-nodes[e])/dz
    return out


@dataclass(frozen=True)
class FEMDepositionSystem:
    z_lower: float
    h: float
    nodes: FloatArray
    deposition_velocity: float
    M: FloatArray
    operator: FloatArray
    rhs_source: FloatArray
    y0: FloatArray
    decay_rates: FloatArray
    eigenvectors: FloatArray

    @property
    def n_elements(self) -> int:
        return int(self.nodes.size-1)

    def coefficients(self, x: float) -> FloatArray:
        if x < 0.0:
            raise ValueError("x must be non-negative")
        amp = self.eigenvectors.T @ (self.M @ self.y0)
        return np.asarray(self.eigenvectors @ (np.exp(-self.decay_rates*float(x))*amp), dtype=np.float64)

    def concentration(self, x: float, z: ArrayLike) -> FloatArray:
        za = np.atleast_1d(np.asarray(z, dtype=np.float64))
        y = self.coefficients(x)
        return np.asarray([_shape_vector(self.nodes, float(zi)) @ y for zi in za], dtype=np.float64)

    def lower_boundary_concentration(self, x: float) -> float:
        return float(self.coefficients(x)[0])

    def flux_derivative_from_constant_test(self, x: float) -> float:
        """Exact finite-dimensional d/dx integral(u*C dz) from partition of unity."""
        y = self.coefficients(x)
        yp = -solve(self.M, self.operator @ y, assume_a="pos", check_finite=True)
        one = np.ones(self.nodes.size, dtype=np.float64)
        return float(one @ (self.M @ yp))


def assemble_fem_deposition_system(
    *,
    z_lower: float,
    h: float,
    n_elements: int,
    wind: Profile,
    diffusivity: Profile,
    source_height: float,
    emission_rate: float,
    deposition_velocity: float,
    grading_power: float = 2.0,
    element_quad_order: int = 4,
) -> FEMDepositionSystem:
    z_lower=float(z_lower); h=float(h); source_height=float(source_height)
    emission_rate=float(emission_rate); vg=float(deposition_velocity)
    if not z_lower <= source_height <= h:
        raise ValueError("source_height outside FEM interval")
    if not np.isfinite(emission_rate) or emission_rate < 0.0:
        raise ValueError("invalid emission_rate")
    if not np.isfinite(vg) or vg < 0.0:
        raise ValueError("deposition_velocity must be finite and >=0")
    if int(element_quad_order) != element_quad_order or element_quad_order < 2:
        raise ValueError("element_quad_order must be integer >=2")

    nodes = graded_nodes(z_lower=z_lower, h=h, n_elements=n_elements, grading_power=grading_power)
    n = nodes.size
    M = np.zeros((n,n), dtype=np.float64)
    A = np.zeros((n,n), dtype=np.float64)
    gx, gw = leggauss(int(element_quad_order))

    for e in range(n-1):
        zl, zu = float(nodes[e]), float(nodes[e+1]); dz=zu-zl
        zq = 0.5*(zl+zu) + 0.5*dz*gx
        wq = 0.5*dz*gw
        N = np.column_stack(((zu-zq)/dz, (zq-zl)/dz))
        dN = np.array([-1.0/dz, 1.0/dz], dtype=np.float64)
        u = np.asarray(wind(zq), dtype=np.float64)
        k = np.asarray(diffusivity(zq), dtype=np.float64)
        if u.shape != zq.shape or k.shape != zq.shape:
            raise ValueError("profile shape mismatch")
        if not np.all(np.isfinite(u)) or np.any(u < 0.0) or not np.any(u > 0.0):
            raise ValueError("wind must be finite, nonnegative, and positive on nonzero measure")
        if not np.all(np.isfinite(k)) or np.any(k < 0.0):
            raise ValueError("diffusivity must be finite and nonnegative")
        Me = N.T @ ((wq*u)[:,None]*N)
        Se = np.outer(dN,dN) * float(np.sum(wq*k))
        ind = np.ix_([e,e+1],[e,e+1])
        M[ind] += Me; A[ind] += Se

    A[0,0] += vg
    M = 0.5*(M+M.T); A = 0.5*(A+A.T)
    me = np.linalg.eigvalsh(M)
    if me[0] <= 100*np.finfo(float).eps*max(1.0,me[-1]):
        raise FloatingPointError("FEM wind mass matrix is not numerically positive definite")
    mu, vecs = eigh(A,M,check_finite=True)
    scale=max(1.0,float(np.max(np.abs(mu)))); tol=500*np.finfo(float).eps*scale
    if np.min(mu) < -tol:
        raise FloatingPointError("materially negative FEM decay rate")
    mu=np.where(np.abs(mu)<=tol,0.0,mu)
    rhs=emission_rate*_shape_vector(nodes,source_height)
    y0=solve(M,rhs,assume_a="pos",check_finite=True)
    return FEMDepositionSystem(
        z_lower=z_lower,h=h,nodes=nodes,deposition_velocity=vg,M=M,operator=A,
        rhs_source=rhs,y0=y0,decay_rates=np.asarray(mu,dtype=np.float64),
        eigenvectors=np.asarray(vecs,dtype=np.float64),
    )
