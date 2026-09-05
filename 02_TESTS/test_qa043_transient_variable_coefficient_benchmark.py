from __future__ import annotations

from functools import lru_cache
import inspect
import math

import numpy as np
import pytest

from gilttpy.numerics.inverse_laplace_modern import dehoog_inverse_laplace
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import (
    assemble_settling_legendre_system,
    assemble_transient_settling_legendre_system,
)
from gilttpy.solvers.transient_2d import fixed_talbot_inverse
from gilttpy.solvers.steady_2d_deposition_legendre import assemble_legendre_boundary_system
from gilttpy.solvers.transient_2d_deposition_legendre import assemble_transient_legendre_boundary_system
from gilttpy.validation.robustness_campaign import GaussianInletProfile
from gilttpy.validation.transient_variable_coefficient import (
    ExactTransientVariableDiffusivityCase,
    assemble_direct_transient_fem_fv_reference,
    spectral_dehoog_concentration_from_inlet,
    spectral_laplace_concentration_from_inlet,
)
from gilttpy.validation.variable_coefficient import (
    ManufacturedVariableCoefficientCase,
    project_profile_to_legendre,
    spectral_solution_from_initial_coefficients,
)


def _rel_l2(got, ref):
    got=np.asarray(got,dtype=float); ref=np.asarray(ref,dtype=float)
    return float(np.linalg.norm(got-ref)/np.linalg.norm(ref))


@lru_cache(maxsize=None)
def _exact_spectral():
    c=ExactTransientVariableDiffusivityCase()
    sys=assemble_settling_legendre_system(
        h=c.h,n_modes=34,wind=c.wind,diffusivity=c.diffusivity,
        source_height=0.5*(c.z_lower+c.h),emission_rate=1.0,
        settling_velocity_m_s=c.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(c.boundary_sink_velocity_m_s,0.0,"QA043 exact causal"),
        n_quad=220,z_lower=c.z_lower,
    )
    y0=project_profile_to_legendre(
        c.inlet_profile,h=c.h,n_modes=sys.n_modes,z_lower=c.z_lower,n_quad=512,
    )
    return c,sys,y0


@lru_cache(maxsize=None)
def _full_variable_spectral():
    c=ManufacturedVariableCoefficientCase()
    inlet=GaussianInletProfile(
        c.z_lower,c.h,center_m=40.0,sigma_m=8.0,source_rate=1.0,
        label="QA043 smooth inlet",provenance="QA043 validation-only smooth inlet",
    )
    sys=assemble_settling_legendre_system(
        h=c.h,n_modes=28,wind=c.wind,diffusivity=c.diffusivity,
        source_height=40.0,emission_rate=1.0,
        settling_velocity_m_s=c.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(c.boundary_sink_velocity_m_s,0.0,"QA043 full variable"),
        n_quad=220,z_lower=c.z_lower,
    )
    profile=inlet.profile(c.wind)
    y0=project_profile_to_legendre(profile,h=c.h,n_modes=sys.n_modes,z_lower=c.z_lower,n_quad=512)
    return c,inlet,profile,sys,y0


def _direct(nx):
    c,inlet,profile,_,_=_full_variable_spectral()
    return assemble_direct_transient_fem_fv_reference(
        z_lower=c.z_lower,h=c.h,x_end=200.0,n_x=nx,n_z_elements=40,
        wind=c.wind,diffusivity=c.diffusivity,
        settling_velocity_m_s=c.settling_velocity_m_s,
        boundary_sink_velocity_m_s=c.boundary_sink_velocity_m_s,
        inlet_profile=profile,element_quad_order=5,
    )


def _full_ref(t):
    c,_,_,sys,y0=_full_variable_spectral()
    z=np.array([15.0,25.0,40.0,55.0,70.0,85.0,105.0])
    values=np.array([
        spectral_dehoog_concentration_from_inlet(
            sys,y0,100.0,float(zz),float(t),degree=26,working_dps=35
        ) for zz in z
    ])
    return z,values


def test_qa043_01_exact_transient_variable_diffusivity_pde_and_boundaries():
    c=ExactTransientVariableDiffusivityCase()
    z=np.linspace(c.z_lower,c.h,101)
    k=c.diffusivity(z)
    assert np.ptp(k) > 0.5
    assert np.all(k > 0.0)
    np.testing.assert_allclose(c.vertical_flux_derivative(z),-c.mu*c.inlet_profile(z),rtol=2e-14,atol=2e-16)
    flux=c.downward_vertical_flux(np.array([c.z_lower,c.h]))
    lower=c.boundary_sink_velocity_m_s*c.inlet_profile([c.z_lower])[0]
    assert flux[0] == pytest.approx(lower,rel=2e-14,abs=2e-16)
    assert abs(flux[1]) < 2e-16
    # inlet advective throughflow is exactly one by construction
    exact=c.wind_m_s*c.inlet_amplitude*(1.0-math.exp(-c.a*c.length))/c.a
    assert exact == pytest.approx(1.0,rel=0,abs=2e-15)


def test_qa043_02_spectral_laplace_matches_exact_delayed_transform():
    c,sys,y0=_exact_spectral()
    worst=0.0
    for x,z,s in ((1.0,1.0,0.2+0.3j),(4.0,3.0,1.0+2.0j),(6.0,8.0,0.05+0.1j)):
        got=spectral_laplace_concentration_from_inlet(sys,y0,x,z,s)
        ref=complex(c.exact_laplace_concentration(x,[z],s)[0])
        worst=max(worst,abs(got-ref)/abs(ref))
    assert worst < 2e-10


def test_qa043_03_dehoog_is_causal_before_exact_advective_arrival():
    c,sys,y0=_exact_spectral()
    x=4.0; z=3.0
    assert c.arrival_time_s(x) == pytest.approx(2.0)
    for t in (1.0,1.5,1.9):
        got=spectral_dehoog_concentration_from_inlet(sys,y0,x,z,t,degree=28,working_dps=40)
        assert abs(got) < 2e-8
        assert c.exact_concentration(x,[z],t)[0] == 0.0


def test_qa043_04_dehoog_postarrival_error_decays_away_from_discontinuous_front():
    c,sys,y0=_exact_spectral()
    x=4.0; z=3.0
    errors=[]
    for t in (2.1,2.2,2.3,2.75,3.0,5.0):
        got=spectral_dehoog_concentration_from_inlet(sys,y0,x,z,t,degree=28,working_dps=40)
        ref=float(c.exact_concentration(x,[z],t)[0])
        errors.append(abs(got-ref)/ref)
    # The transform is discontinuous at t=2.0.  The near-front layer is
    # characterized rather than hidden; strong accuracy is required once away
    # from that discontinuity.
    assert errors[0] < 2e-4
    assert errors[1] < 1e-5
    assert errors[2] < 1e-6
    assert max(errors[3:]) < 1e-8
    assert errors[2] < errors[1] < errors[0]


def test_qa043_05_fixed_talbot_delayed_front_failure_is_explicit_not_hidden():
    c=ExactTransientVariableDiffusivityCase(); x=4.0; z=3.0
    f=lambda s: complex(c.exact_laplace_concentration(x,[z],s)[0])
    # Known Fixed-Talbot limitation: the delayed step is zero at t=1 but the
    # left-bending contour produces a catastrophic noncausal value.
    historical=fixed_talbot_inverse(f,1.0,mstar=9)
    modern=dehoog_inverse_laplace(f,1.0,degree=28,working_dps=40)
    assert abs(historical) > 1e3
    assert abs(modern) < 1e-10


def test_qa043_06_modern_settling_solver_exposes_dehoog_and_reaches_steady_reference():
    wind=lambda z:np.full_like(np.asarray(z,dtype=float),2.0)
    diff=lambda z:np.full_like(np.asarray(z,dtype=float),0.6)
    kwargs=dict(
        h=6.0,n_modes=20,wind=wind,diffusivity=diff,source_height=3.0,
        emission_rate=1.0,settling_velocity_m_s=0.01,
        boundary=LinearRobinBoundaryCondition(0.02,0.0,"QA043 modern inversion"),
        n_quad=100,z_lower=0.0,
    )
    steady=assemble_settling_legendre_system(**kwargs)
    transient=assemble_transient_settling_legendre_system(**kwargs)
    got=transient.concentration_dehoog(3.0,2.0,3.0,degree=26,working_dps=35)
    ref=float(steady.concentration(3.0,np.array([2.0]))[0])
    assert abs(got-ref)/abs(ref) < 2e-8

    # The same modern inversion provider is wired into the QA-037 no-settling
    # typed-boundary transient system, without changing the historical core.
    base_kwargs=dict(h=6.0,n_modes=20,wind=wind,diffusivity=diff,source_height=3.0,
                     emission_rate=1.0,boundary=LinearRobinBoundaryCondition(0.02,0.0,"QA043 no settling"),
                     n_quad=100,z_lower=0.0)
    steady0=assemble_legendre_boundary_system(**base_kwargs)
    transient0=assemble_transient_legendre_boundary_system(**base_kwargs)
    got0=transient0.concentration_dehoog(3.0,2.0,3.0,degree=26,working_dps=35)
    ref0=float(steady0.concentration(3.0,np.array([2.0]))[0])
    assert abs(got0-ref0)/abs(ref0) < 2e-8


def test_qa043_07_full_variable_coefficients_dehoog_agree_with_direct_time_fem_fv():
    z,ref=_full_ref(60.0)
    got=_direct(100).concentration(100.0,z,60.0)
    assert _rel_l2(got,ref) < 2e-3


def test_qa043_08_direct_time_reference_refines_toward_spectral_dehoog():
    z,ref=_full_ref(100.0)
    e50=_rel_l2(_direct(50).concentration(100.0,z,100.0),ref)
    e100=_rel_l2(_direct(100).concentration(100.0,z,100.0),ref)
    e200=_rel_l2(_direct(200).concentration(100.0,z,100.0),ref)
    assert e100 < 0.35*e50
    assert e200 < e100
    assert e200 < 1e-3


def test_qa043_09_dehoog_long_time_recovers_full_variable_steady_state():
    c,_,_,sys,y0=_full_variable_spectral()
    z=np.array([15.0,25.0,40.0,55.0,70.0,85.0,105.0])
    got=np.array([
        spectral_dehoog_concentration_from_inlet(sys,y0,100.0,float(zz),80.0,degree=28,working_dps=40)
        for zz in z
    ])
    ref=spectral_solution_from_initial_coefficients(sys,y0,100.0,z)
    assert _rel_l2(got,ref) < 1e-7


def test_qa043_10_scope_and_provenance_guards_preserve_historical_fixed_talbot():
    import gilttpy.validation.transient_variable_coefficient as vmod
    import gilttpy.numerics.inverse_laplace_modern as imod
    import gilttpy.solvers.transient_2d as historical
    src=(inspect.getsource(vmod)+inspect.getsource(imod)).lower()
    assert "copenhagen" not in src and "hanford" not in src and "target tuning" in src
    assert hasattr(historical,"fixed_talbot_inverse")
    with pytest.raises(ValueError):
        dehoog_inverse_laplace(lambda s:1.0/s,0.0)
    with pytest.raises(ValueError):
        dehoog_inverse_laplace(lambda s:1.0/s,1.0,degree=4)
