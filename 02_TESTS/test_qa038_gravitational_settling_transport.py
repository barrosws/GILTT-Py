from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from scipy.linalg import expm, solve

from gilttpy.basis.shifted_legendre import lower_values, top_values
from gilttpy.physics.aerosol_transport_coupling import (
    effective_velocity_above_resolved_resistance,
    venkatram_pleim_resolved_interface_flux_law,
    zhang2001_split_resolved_interface_flux_law,
)
from gilttpy.physics.deposition import AerosolResistanceSettling
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import (
    assemble_settling_legendre_system,
    assemble_transient_settling_legendre_system,
)
from gilttpy.solvers.steady_2d_deposition_legendre import assemble_legendre_boundary_system
from gilttpy.solvers.transient_2d_deposition_legendre import assemble_transient_legendre_boundary_system


def _profiles(u=2.0, k=0.7):
    wind=lambda z: np.full_like(np.asarray(z,dtype=float),u,dtype=float)
    diff=lambda z: np.full_like(np.asarray(z,dtype=float),k,dtype=float)
    return wind,diff


def _system(vg=0.012,kb=0.025,n=28):
    wind,diff=_profiles()
    boundary=LinearRobinBoundaryCondition(kb,0.0,"qa038 structural total flux")
    return assemble_settling_legendre_system(
        h=8.0,n_modes=n,wind=wind,diffusivity=diff,source_height=3.1,
        emission_rate=1.0,settling_velocity_m_s=vg,boundary=boundary,
        n_quad=120,z_lower=0.0,
    )


def test_01_settling_operator_has_exact_conservative_integration_by_parts_identity():
    vg=0.017
    sys=_system(vg=vg,n=24)
    b=lower_values(h=sys.h,n_modes=sys.n_modes,z_lower=sys.z_lower)
    t=top_values(h=sys.h,n_modes=sys.n_modes,z_lower=sys.z_lower)
    expected=vg*(np.outer(t,t)-np.outer(b,b))
    np.testing.assert_allclose(sys.settling_operator+sys.settling_operator.T,expected,rtol=2e-12,atol=2e-12)
    # Constant test has zero derivative, hence settling contributes no direct
    # volume sink to total resolved mass.
    assert np.linalg.norm(sys.settling_operator[0]) < 3e-14


def test_02_zero_settling_exactly_reduces_to_qa037_homogeneous_operator_and_solution():
    wind,diff=_profiles(2.3,0.8)
    boundary=LinearRobinBoundaryCondition(0.018,0.0,"same")
    kwargs=dict(h=9.0,n_modes=22,wind=wind,diffusivity=diff,source_height=3.2,
                emission_rate=1.1,n_quad=90,z_lower=0.2,boundary=boundary)
    old=assemble_legendre_boundary_system(**kwargs)
    new=assemble_settling_legendre_system(**kwargs,settling_velocity_m_s=0.0)
    np.testing.assert_array_equal(new.M,old.M)
    np.testing.assert_array_equal(new.diffusion_operator,old.diffusion_operator)
    np.testing.assert_array_equal(new.boundary_matrix,old.boundary_matrix)
    np.testing.assert_array_equal(new.operator,old.operator)
    np.testing.assert_array_equal(new.y0,old.y0)
    for x in (0.0,0.5,3.0):
        np.testing.assert_allclose(new.coefficients(x),old.coefficients_expm(x),rtol=0,atol=0)


def test_03_constant_test_mass_balance_counts_total_boundary_flux_once():
    sys=_system(vg=0.014,kb=0.031,n=34)
    for x in (0.2,1.0,4.0):
        dflux=sys.flux_derivative_from_weak_constant_test(x)
        j=sys.lower_boundary_downward_flux(x)
        assert dflux == pytest.approx(-j,rel=3e-11,abs=3e-11)


def test_04_vp_subinterface_flux_is_exactly_compositional_across_resistance_segments():
    for vg in (0.0,1e-5,0.01,0.08):
        rr=40.0; ru=60.0
        law=venkatram_pleim_resolved_interface_flux_law(
            settling_velocity_m_s=vg,unresolved_resistance_s_m=ru,
            provenance="QA038 target-free VP composition",
        )
        composed=effective_velocity_above_resolved_resistance(
            settling_velocity_m_s=vg,
            lower_total_exit_velocity_m_s=law.sink_velocity_m_s,
            resolved_resistance_s_m=rr,
        )
        direct=AerosolResistanceSettling(0.0,rr+ru,vg).deposition_velocity()
        assert composed == pytest.approx(direct,rel=3e-14,abs=3e-14)


def test_05_zhang_split_is_local_total_flux_and_not_falsely_declared_partition_invariant():
    vg=0.01; rr=40.0; ra_sub=20.0; rs=40.0
    law=zhang2001_split_resolved_interface_flux_law(
        settling_velocity_m_s=vg,residual_aerodynamic_resistance_s_m=ra_sub,
        surface_resistance_s_m=rs,provenance="QA038 Z01 split audit",
    )
    assert law.sink_velocity_m_s == pytest.approx(vg+1.0/(ra_sub+rs),rel=0,abs=1e-15)
    effective=effective_velocity_above_resolved_resistance(
        settling_velocity_m_s=vg,lower_total_exit_velocity_m_s=law.sink_velocity_m_s,
        resolved_resistance_s_m=rr,
    )
    original_reference_additive=vg+1.0/(rr+ra_sub+rs)
    # A nonzero resolved settling segment changes the concentration profile;
    # pretending the original additive reference-height formula is invariant
    # would silently double/reshape transport.
    assert abs(effective-original_reference_additive)/original_reference_additive > 0.05
    # In the no-settling limit, ordinary resistance additivity is recovered.
    law0=zhang2001_split_resolved_interface_flux_law(
        settling_velocity_m_s=0.0,residual_aerodynamic_resistance_s_m=ra_sub,
        surface_resistance_s_m=rs,provenance="QA038 Z01 zero-settling audit",
    )
    eff0=effective_velocity_above_resolved_resistance(
        settling_velocity_m_s=0.0,lower_total_exit_velocity_m_s=law0.sink_velocity_m_s,
        resolved_resistance_s_m=rr,
    )
    assert eff0 == pytest.approx(1.0/(rr+ra_sub+rs),rel=2e-15,abs=2e-15)


def test_06_vp_zero_and_strong_settling_limits_are_correct():
    r=80.0
    zero=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=0.0,unresolved_resistance_s_m=r,provenance="limits"
    )
    assert zero.sink_velocity_m_s == pytest.approx(1.0/r,rel=0,abs=1e-15)
    strong=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=0.5,unresolved_resistance_s_m=r,provenance="limits"
    )
    assert strong.sink_velocity_m_s/0.5 == pytest.approx(1.0,rel=1e-14,abs=1e-14)
    pure=zhang2001_split_resolved_interface_flux_law(
        settling_velocity_m_s=0.02,residual_aerodynamic_resistance_s_m=0.0,
        surface_resistance_s_m=math.inf,provenance="pure settling"
    )
    assert pure.sink_velocity_m_s == 0.02
    assert pure.nonsettling_increment_m_s == 0.0


def _fem_reference(*,h,ne,u,k,vg,kb,hs,q,x):
    nodes=np.linspace(0.0,h,ne+1)
    n=nodes.size
    M=np.zeros((n,n)); A=np.zeros((n,n))
    # 4-point Gauss rule gives ample accuracy for constant coefficients/linear FEM.
    xi,wi=np.polynomial.legendre.leggauss(4)
    for e in range(ne):
        z0,z1=nodes[e],nodes[e+1]; le=z1-z0
        ids=(e,e+1)
        for xx,ww in zip(xi,wi):
            N=np.array([(1-xx)/2,(1+xx)/2])
            dN=np.array([-1/le,1/le])
            jac=le/2
            for a in range(2):
                for b in range(2):
                    M[ids[a],ids[b]] += ww*jac*u*N[a]*N[b]
                    A[ids[a],ids[b]] += ww*jac*(k*dN[a]*dN[b] + vg*dN[a]*N[b])
    A[0,0]+=kb
    # consistent point source at inlet x=0
    rhs=np.zeros(n)
    j=np.searchsorted(nodes,hs)-1; j=max(0,min(j,ne-1))
    le=nodes[j+1]-nodes[j]; a=(nodes[j+1]-hs)/le; b=(hs-nodes[j])/le
    rhs[j]+=q*a; rhs[j+1]+=q*b
    y0=solve(M,rhs,assume_a="pos")
    gen=-solve(M,A,assume_a="gen")
    return nodes,expm(gen*x)@y0


def _fem_eval(nodes,y,z):
    out=[]
    for zz in np.atleast_1d(z):
        j=np.searchsorted(nodes,zz)-1; j=max(0,min(j,len(nodes)-2))
        le=nodes[j+1]-nodes[j]
        a=(nodes[j+1]-zz)/le; b=(zz-nodes[j])/le
        out.append(a*y[j]+b*y[j+1])
    return np.asarray(out)


def test_07_settling_legendre_agrees_with_independent_piecewise_linear_fem():
    h=6.0; u=1.8; k=0.6; vg=0.018; kb=0.032; hs=2.4; q=1.0; x=1.5
    wind,diff=_profiles(u,k)
    leg=assemble_settling_legendre_system(
        h=h,n_modes=44,wind=wind,diffusivity=diff,source_height=hs,
        emission_rate=q,settling_velocity_m_s=vg,
        boundary=LinearRobinBoundaryCondition(kb,0.0,"FEM crosscheck"),
        n_quad=180,z_lower=0.0,
    )
    nodes,yf=_fem_reference(h=h,ne=300,u=u,k=k,vg=vg,kb=kb,hs=hs,q=q,x=x)
    z=np.array([0.0,0.5,1.5,3.0,5.5])
    cf=_fem_eval(nodes,yf,z); cl=leg.concentration(x,z)
    rel=np.linalg.norm(cl-cf)/np.linalg.norm(cf)
    assert rel < 2.5e-3


def test_08_transient_zero_settling_exactly_reduces_to_qa037_laplace_solver():
    wind,diff=_profiles(2.1,0.75)
    boundary=LinearRobinBoundaryCondition(0.02,0.0,"zero-settling transient")
    kwargs=dict(h=7.0,n_modes=18,wind=wind,diffusivity=diff,source_height=2.8,
                emission_rate=1.0,n_quad=80,z_lower=0.1,boundary=boundary)
    old=assemble_transient_legendre_boundary_system(**kwargs)
    new=assemble_transient_settling_legendre_system(**kwargs,settling_velocity_m_s=0.0)
    for s in (0.2+0.4j,1.1+0.3j):
        np.testing.assert_allclose(new.laplace_coefficients(1.2,s),old.laplace_coefficients(1.2,s),rtol=0,atol=0)


def test_09_transient_final_value_recovers_steady_settling_solution():
    vg=0.015
    law=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=vg,unresolved_resistance_s_m=50.0,provenance="final-value"
    )
    wind,diff=_profiles(2.0,0.65)
    kwargs=dict(h=7.5,n_modes=24,wind=wind,diffusivity=diff,source_height=3.0,
                emission_rate=1.2,n_quad=100,z_lower=0.0,
                settling_velocity_m_s=vg,boundary=law)
    steady=assemble_settling_legendre_system(**kwargs)
    transient=assemble_transient_settling_legendre_system(**kwargs)
    s=1e-8
    for x,z in ((0.5,0.3),(1.5,2.0),(4.0,6.0)):
        cfinal=s*transient.laplace_concentration(x,z,s)
        csteady=float(steady.concentration(x,np.array([z]))[0])
        assert cfinal.real == pytest.approx(csteady,rel=3e-7,abs=3e-8)
        assert abs(cfinal.imag) < 1e-11


def test_10_scope_guards_no_affine_particle_emission_no_complete_closure_import_and_historical_untouched():
    wind,diff=_profiles()
    with pytest.raises(ValueError):
        assemble_settling_legendre_system(
            h=6,n_modes=12,wind=wind,diffusivity=diff,source_height=2,
            emission_rate=1,settling_velocity_m_s=0.01,
            boundary=LinearRobinBoundaryCondition(0.02,1.0,"not aerosol"),n_quad=50,
        )
    import gilttpy.solvers.settling_2d_legendre as smod
    src=inspect.getsource(smod)
    assert "aerosol_deposition" not in src
    assert "aerosol_size_distribution" not in src
    assert "historical" in src.lower()
