"""Transient 2-D Fickian GILTT core for Buske Eq. (3.50).


The time variable is transformed by Laplace. For w=0 and zero initial
concentration, the unnormalised cosine-Galerkin system is


    M dY/dx + (S + s G) Y = 0,
    M Y(0,s) = (Q/s) psi(Hs),


where M_ij=int u psi_i psi_j dz, S_ij=int Kz psi'_i psi'_j dz and
G_ij=int psi_i psi_j dz. This is the conservative weak-form equivalent of
the transformed equation used in the Buske/Moreira GILTT lineage.


The historical Fixed-Talbot inversion is implemented explicitly from Buske
Eq. (3.47). For the older Gaussian inverse-Laplace formula, the M=2 roots
and weights are now lineage-source confirmed from a Vilhena coauthored paper
that reproduces the Stroud-Secrest complex Gaussian table explicitly. Literal
Buske-2008 Fortran assignment remains a narrower provenance question.
"""
from __future__ import annotations


from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, solve


from gilttpy.basis.cosine_neumann import values, derivatives
from gilttpy.basis.quadrature import gauss_legendre_interval
from gilttpy.numerics.inverse_laplace_gaussian import (
    GaussianInverseRule,
    gaussian_inverse_laplace,
    stroud_secrest_m2_lineage_rule,
)


FloatArray = NDArray[np.float64]
Profile = Callable[[FloatArray], ArrayLike]




def fixed_talbot_inverse(laplace_fn: Callable[[complex], complex], t: float, *, mstar: int=5) -> float:
    """Invert a scalar Laplace transform using Buske Eq. (3.47).


    r* = 2 M*/(5 t), theta_k=k*pi/M*,
    S(theta)=r* theta (cot(theta)+i),
    sigma(theta)=theta+(theta*cot(theta)-1)*cot(theta).
    """
    t=float(t)
    if not np.isfinite(t) or t<=0:
        raise ValueError("t must be finite and positive")
    if int(mstar)!=mstar or mstar<2:
        raise ValueError("mstar must be an integer >=2")
    mstar=int(mstar)
    rstar=2.0*mstar/(5.0*t)
    total=0.5*np.exp(rstar*t)*complex(laplace_fn(complex(rstar,0.0)))
    for k in range(1,mstar):
        theta=k*np.pi/mstar
        cot=1.0/np.tan(theta)
        s=rstar*theta*(cot+1j)
        sigma=theta+(theta*cot-1.0)*cot
        term=np.exp(t*s)*complex(laplace_fn(complex(s)))*(1.0+1j*sigma)
        total += np.real(term)
    out=(rstar/mstar)*total
    return float(np.real(out))




@dataclass(frozen=True)
class Transient2DSystem:
    h: float
    n_modes: int
    M: FloatArray
    S: FloatArray
    G: FloatArray
    psi_source: FloatArray
    emission_rate: float


    def laplace_coefficients(self, x: float, s: complex) -> NDArray[np.complex128]:
        """Return modal coefficients Y(x,s) of the Laplace-transformed solution."""
        x=float(x); s=complex(s)
        if not np.isfinite(x) or x<0:
            raise ValueError("x must be finite and non-negative")
        if not np.isfinite(s.real) or not np.isfinite(s.imag) or abs(s)==0:
            raise ValueError("s must be finite and nonzero")
        Mc=self.M.astype(np.complex128,copy=False)
        Ac=self.S.astype(np.complex128,copy=False)+s*self.G.astype(np.complex128,copy=False)
        rhs=(self.emission_rate/s)*self.psi_source.astype(np.complex128,copy=False)
        y0=solve(Mc,rhs,assume_a="gen",check_finite=True)
        generator=-solve(Mc,Ac,assume_a="gen",check_finite=True)
        return expm(generator*x)@y0


    def laplace_concentration(self, x: float, z: float, s: complex) -> complex:
        """Return Cbar(x,z,s)."""
        z=float(z)
        if not np.isfinite(z) or z<0 or z>self.h:
            raise ValueError("z outside [0,h]")
        psi=values(np.asarray([z]),self.h,self.n_modes)[0]
        return complex(psi@self.laplace_coefficients(x,s))


    def concentration_fixed_talbot(self, x: float, z: float, t: float, *, mstar: int=5) -> float:
        """Invert Cbar(x,z,s) using the historical Fixed-Talbot formula."""
        return fixed_talbot_inverse(lambda s:self.laplace_concentration(x,z,s),t,mstar=mstar)


    def ground_concentration_fixed_talbot(self, x: float, t: float, *, mstar: int=5) -> float:
        return self.concentration_fixed_talbot(x,0.0,t,mstar=mstar)


    def concentration_gaussian_inverse(self, x: float, z: float, t: float, *,
                                       roots: ArrayLike, weights: ArrayLike) -> float:
        """Evaluate Buske Eq. (3.45) for explicitly supplied P_k/A_k arrays."""
        p=np.asarray(roots,dtype=np.complex128)
        a=np.asarray(weights,dtype=np.complex128)
        if p.ndim!=1 or a.ndim!=1 or p.size==0 or p.shape!=a.shape:
            raise ValueError("roots and weights must be nonempty 1-D arrays of equal length")
        rule=GaussianInverseRule(
            order=int(p.size), roots=p, weights=a,
            rule_id="explicit_user_supplied", provenance_class="explicit_external_input",
        )
        return gaussian_inverse_laplace(
            lambda s:self.laplace_concentration(x,z,s), t, rule=rule
        )

    def concentration_gaussian_m2_lineage(self, x: float, z: float, t: float) -> float:
        """Evaluate the lineage-source-confirmed historical M=2 rule.

        QA-013 classifies M=2 as a historical/reproduction comparator rather
        than the numerical-accuracy default for Copenhagen transient results.
        """
        rule=stroud_secrest_m2_lineage_rule()
        return gaussian_inverse_laplace(
            lambda s:self.laplace_concentration(x,z,s), t, rule=rule
        )

    def concentration_gaussian_m2_reconstructed(self, x: float, z: float, t: float) -> float:
        """Backward-compatible alias for the source-confirmed M=2 rule."""
        return self.concentration_gaussian_m2_lineage(x,z,t)




def assemble_transient_system(*, h: float, n_modes: int, wind: Profile,
                              diffusivity: Profile, source_height: float,
                              emission_rate: float, n_quad: int=256) -> Transient2DSystem:
    """Assemble the Laplace-domain matrices for Buske Eq. (3.50), w=0."""
    h=float(h); source_height=float(source_height); emission_rate=float(emission_rate)
    if not np.isfinite(h) or h<=0:
        raise ValueError("h must be finite and positive")
    if n_modes<1:
        raise ValueError("n_modes must be >=1")
    if n_quad<max(2,n_modes):
        raise ValueError("n_quad too small")
    if not 0<=source_height<=h:
        raise ValueError("source height outside [0,h]")
    if not np.isfinite(emission_rate) or emission_rate<0:
        raise ValueError("invalid emission_rate")
    zq,wq=gauss_legendre_interval(0.0,h,int(n_quad))
    u=np.asarray(wind(zq),dtype=np.float64)
    k=np.asarray(diffusivity(zq),dtype=np.float64)
    if u.shape!=zq.shape or k.shape!=zq.shape:
        raise ValueError("profile shape mismatch")
    if not np.all(np.isfinite(u)) or np.any(u<0) or not np.any(u>0):
        raise ValueError("wind must be finite, nonnegative, and positive on a set of nonzero measure")
    if not np.all(np.isfinite(k)) or np.any(k<0):
        raise ValueError("diffusivity must be finite and nonnegative")
    psi=values(zq,h,int(n_modes)); dpsi=derivatives(zq,h,int(n_modes))
    M=psi.T@((wq*u)[:,None]*psi)
    S=dpsi.T@((wq*k)[:,None]*dpsi)
    G=psi.T@((wq)[:,None]*psi)
    M=0.5*(M+M.T); S=0.5*(S+S.T); G=0.5*(G+G.T)
    me=np.linalg.eigvalsh(M)
    if me[0]<=100*np.finfo(float).eps*max(1.0,me[-1]):
        raise FloatingPointError("wind mass matrix is not numerically positive definite")
    psi_source=values(np.asarray([source_height]),h,int(n_modes))[0]
    return Transient2DSystem(h,int(n_modes),M,S,G,psi_source,emission_rate)