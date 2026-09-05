from __future__ import annotations
import inspect
import math
import numpy as np
import pytest

from gilttpy.numerics.inverse_laplace_policy import ModernInverseLaplacePolicy, InverseLaplaceMethod
from gilttpy.numerics.inverse_laplace_modern import dehoog_inverse_laplace
from gilttpy.solvers.transient_2d import fixed_talbot_inverse
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.inverse_laplace_robustness import (
    exponential_case, delayed_step_case, delayed_exponential_case,
    damped_sine_case, damped_cosine_case, diffusion_erfc_case,
)
from gilttpy.validation.transient_variable_coefficient import (
    ExactTransientVariableDiffusivityCase, spectral_dehoog_concentration_from_inlet,
)
from gilttpy.validation.variable_coefficient import project_profile_to_legendre


def _rel(got, ref):
    return abs(got-ref)/max(abs(ref), 1e-14)


def test_qa044_01_modern_policy_is_explicit_dehoog_only_and_source_tagged():
    p=ModernInverseLaplacePolicy()
    assert p.method is InverseLaplaceMethod.DEHOOG
    assert "de hoog" in p.provenance.lower()
    assert "target-free" in p.provenance.lower()
    with pytest.raises(ValueError):
        ModernInverseLaplacePolicy(degree=4)


def test_qa044_02_smooth_exponential_accuracy_across_time_decades():
    c=exponential_case(); p=ModernInverseLaplacePolicy()
    worst=0.0
    for t in (0.02,0.1,1.0,5.0,20.0):
        worst=max(worst,_rel(p.invert(c.laplace,t),c.exact(t)))
    assert max(abs(p.invert(c.laplace,t)-c.exact(t)) for t in (0.02,0.1,1.0,5.0,20.0)) < 2e-9


def test_qa044_03_delayed_step_causality_and_postfront_accuracy():
    c=delayed_step_case(2.0); p=ModernInverseLaplacePolicy()
    for t in (0.5,1.0,1.8):
        assert abs(p.invert(c.laplace,t)) < 2e-9
    # avoid the discontinuity itself; characterize the smooth sides.
    for t in (2.4,3.0,8.0):
        assert _rel(p.invert(c.laplace,t),1.0) < 2e-7


def test_qa044_04_delayed_exponential_preserves_causality_and_decay():
    c=delayed_exponential_case(); p=ModernInverseLaplacePolicy()
    for t in (0.5,1.0,1.4):
        assert abs(p.invert(c.laplace,t)) < 2e-8
    worst=max(_rel(p.invert(c.laplace,t),c.exact(t)) for t in (1.8,2.5,5.0,12.0))
    assert worst < 3e-6


def test_qa044_05_damped_sine_oscillatory_accuracy():
    c=damped_sine_case(); p=ModernInverseLaplacePolicy()
    pts=(0.2,0.7,1.4,3.0,6.0)
    err=max(abs(p.invert(c.laplace,t)-c.exact(t)) for t in pts)
    assert err < 2e-7


def test_qa044_06_damped_cosine_oscillatory_accuracy():
    c=damped_cosine_case(); p=ModernInverseLaplacePolicy()
    pts=(0.15,0.6,1.1,2.5,5.0)
    err=max(abs(p.invert(c.laplace,t)-c.exact(t)) for t in pts)
    assert err < 3e-7


def test_qa044_07_diffusion_branch_point_erfc_accuracy():
    c=diffusion_erfc_case(); p=ModernInverseLaplacePolicy()
    worst=max(_rel(p.invert(c.laplace,t),c.exact(t)) for t in (0.1,0.3,1.0,4.0,15.0))
    assert worst < 3e-7


def test_qa044_08_complex128_precision_ceiling_is_explicit_and_empirical():
    c=damped_cosine_case(); t=1.1
    vals=[dehoog_inverse_laplace(c.laplace,t,degree=d,working_dps=40) for d in (22,24,26,28)]
    ref=c.exact(t)
    assert max(_rel(v,ref) for v in vals) < 2e-7
    # Higher arbitrary-precision inversion degree cannot create precision absent
    # from complex128 Laplace-model evaluations; guard this explicitly.
    with pytest.raises(ValueError):
        ModernInverseLaplacePolicy(degree=34)


def test_qa044_09_variable_coefficient_operator_is_causal_over_multiple_distances():
    c=ExactTransientVariableDiffusivityCase()
    sys=assemble_settling_legendre_system(
        h=c.h,n_modes=34,wind=c.wind,diffusivity=c.diffusivity,
        source_height=5.0,emission_rate=1.0,settling_velocity_m_s=c.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(c.boundary_sink_velocity_m_s,0.0,"QA044 causal"),
        n_quad=220,z_lower=c.z_lower)
    y0=project_profile_to_legendre(c.inlet_profile,h=c.h,n_modes=sys.n_modes,z_lower=c.z_lower,n_quad=512)
    for x in (1.0,4.0,8.0):
        tau=c.arrival_time_s(x); z=3.0
        pre=spectral_dehoog_concentration_from_inlet(sys,y0,x,z,0.75*tau,degree=30,working_dps=42)
        assert abs(pre) < 2e-8
        t=tau+max(0.75,0.25*tau)
        got=spectral_dehoog_concentration_from_inlet(sys,y0,x,z,t,degree=30,working_dps=42)
        ref=float(c.exact_concentration(x,[z],t)[0])
        assert _rel(got,ref) < 3e-6


def test_qa044_10_fixed_talbot_is_legacy_explicit_and_not_silently_selected():
    c=delayed_step_case(2.0)
    legacy=fixed_talbot_inverse(c.laplace,1.0,mstar=9)
    modern=ModernInverseLaplacePolicy().invert(c.laplace,1.0)
    assert abs(legacy) > 1e3
    assert abs(modern) < 2e-8
    import gilttpy.numerics.inverse_laplace_policy as pol
    src=inspect.getsource(pol).lower()
    assert "fixed_talbot" not in src
    assert "copenhagen" not in src and "hanford" not in src
