from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh, expm, solve

from gilttpy.basis.shifted_legendre import lower_values
from gilttpy.physics.boundary_coupling import LinearResolvedInterfaceFluxLaw
from gilttpy.solvers.lower_boundary_operator import (
    LinearRobinBoundaryCondition,
    LowerBoundaryFluxLaw,
    coerce_boundary_weak_terms,
)
from gilttpy.solvers.steady_2d_deposition_fem import (
    _shape_vector,
    assemble_fem_deposition_system,
)
from gilttpy.solvers.steady_2d_deposition_legendre import (
    assemble_legendre_boundary_system,
    assemble_legendre_deposition_system,
)
from gilttpy.solvers.transient_2d_deposition_legendre import (
    assemble_transient_legendre_boundary_system,
    assemble_transient_legendre_deposition_system,
)


def _constant_profiles(u: float = 2.0, k: float = 0.7):
    wind = lambda z: np.full_like(np.asarray(z, dtype=float), u, dtype=float)
    diff = lambda z: np.full_like(np.asarray(z, dtype=float), k, dtype=float)
    return wind, diff


def _steady(boundary, *, n_modes=28, n_quad=120, h=10.0, hs=4.0, q=1.3):
    wind, diff = _constant_profiles()
    return assemble_legendre_boundary_system(
        h=h, n_modes=n_modes, wind=wind, diffusivity=diff,
        source_height=hs, emission_rate=q, boundary=boundary,
        n_quad=n_quad, z_lower=0.0,
    )


def test_01_typed_contract_and_qa036_physics_law_generate_exact_B_f():
    b = np.array([1.0, -2.0, 0.5])
    native = LinearRobinBoundaryCondition(0.03, 4.5, "native")
    physics = LinearResolvedInterfaceFluxLaw(0.03, 4.5, "qa036")
    assert isinstance(native, LowerBoundaryFluxLaw)
    assert isinstance(physics, LowerBoundaryFluxLaw)
    for law in (native, physics):
        terms = coerce_boundary_weak_terms(law, b)
        np.testing.assert_allclose(terms.matrix, 0.03*np.outer(b,b), rtol=0, atol=0)
        np.testing.assert_allclose(terms.forcing, 0.03*4.5*b, rtol=0, atol=0)
        assert law.downward_flux(7.0) == pytest.approx(0.03*(7.0-4.5), rel=0, abs=1e-15)


def test_02_typed_homogeneous_steady_is_exact_legacy_wrapper_reduction():
    wind, diff = _constant_profiles(2.3, 0.8)
    kwargs = dict(h=9.0, n_modes=22, wind=wind, diffusivity=diff,
                  source_height=3.2, emission_rate=1.1, n_quad=90, z_lower=0.2)
    legacy = assemble_legendre_deposition_system(**kwargs, deposition_velocity=0.018)
    typed = assemble_legendre_boundary_system(
        **kwargs, boundary=LinearRobinBoundaryCondition(0.018, 0.0, "typed-zero-comp")
    )
    for name in ("M","diffusion_operator","boundary_matrix","boundary_forcing","operator","rhs_source","y0","decay_rates"):
        np.testing.assert_array_equal(getattr(legacy,name), getattr(typed,name))
    for x in (0.0, 0.3, 2.0, 8.0):
        np.testing.assert_allclose(legacy.coefficients(x), typed.coefficients(x), rtol=0, atol=0)


def test_03_zero_flux_typed_boundary_preserves_neumann_and_rejects_inconsistent_affine_state():
    system = _steady(LinearRobinBoundaryCondition(0.0, 0.0, "no-flux"), n_modes=18)
    assert system.decay_rates[0] == 0.0
    assert np.linalg.norm(system.boundary_matrix, ord=np.inf) == 0.0
    assert np.linalg.norm(system.boundary_forcing, ord=np.inf) == 0.0
    with pytest.raises(ValueError):
        LinearRobinBoundaryCondition(0.0, 1.0)


def test_04_constant_compensation_equilibrium_is_exact_discrete_steady_state():
    ceq = 3.25
    h = 8.0
    system = _steady(LinearRobinBoundaryCondition(0.04, ceq, "equilibrium"), h=h, n_modes=30)
    y_const = np.zeros(system.n_modes)
    y_const[0] = ceq*math.sqrt(h)
    for x in (0.0, 0.25, 2.0, 20.0):
        y = system.propagate_coefficients(y_const, x)
        np.testing.assert_allclose(y, y_const, rtol=3e-12, atol=3e-12)
    residual = system.operator@y_const-system.boundary_forcing
    assert np.linalg.norm(residual) < 2e-11


def test_05_affine_steady_eig_expm_and_augmented_matrix_exponential_agree():
    system = _steady(LinearRobinBoundaryCondition(0.025, 2.0, "affine"), n_modes=20)
    x = 1.7
    ye = system.coefficients_eig(x)
    yx = system.coefficients_expm(x)
    np.testing.assert_allclose(ye, yx, rtol=2e-12, atol=2e-12)

    G = -solve(system.M, system.operator, assume_a="pos")
    q = solve(system.M, system.boundary_forcing, assume_a="pos")
    aug = np.zeros((system.n_modes+1, system.n_modes+1))
    aug[:-1,:-1] = G
    aug[:-1,-1] = q
    state0 = np.r_[system.y0, 1.0]
    y_ref = (expm(aug*x)@state0)[:-1]
    np.testing.assert_allclose(ye, y_ref, rtol=3e-12, atol=3e-12)


def test_06_affine_steady_constant_test_mass_balance_equals_signed_boundary_flux():
    system = _steady(LinearRobinBoundaryCondition(0.035, 1.4, "mass-balance"), n_modes=32)
    for x in (0.2, 1.0, 5.0):
        derivative = system.flux_derivative_from_weak_constant_test(x)
        downward = system.lower_boundary_downward_flux(x)
        assert derivative == pytest.approx(-downward, rel=2e-11, abs=2e-11)


def test_07_affine_legendre_agrees_with_independent_piecewise_linear_fem_reference():
    h=6.0; hs=2.4; qsrc=1.0; k=0.03; ceq=1.2
    wind,diff=_constant_profiles(1.8,0.6)
    leg=assemble_legendre_boundary_system(
        h=h,n_modes=44,wind=wind,diffusivity=diff,source_height=hs,
        emission_rate=qsrc,boundary=LinearRobinBoundaryCondition(k,ceq,"cross-FEM"),
        n_quad=160,z_lower=0.0,
    )
    fem=assemble_fem_deposition_system(
        z_lower=0.0,h=h,n_elements=220,wind=wind,diffusivity=diff,
        source_height=hs,emission_rate=qsrc,deposition_velocity=k,
        grading_power=1.0,element_quad_order=4,
    )
    f=np.zeros(fem.nodes.size); f[0]=k*ceq
    y_eq=solve(fem.operator,f,assume_a="pos")
    amp=fem.eigenvectors.T@(fem.M@(fem.y0-y_eq))
    x=1.5
    yf=y_eq+fem.eigenvectors@(np.exp(-fem.decay_rates*x)*amp)
    z=np.array([0.0,0.5,1.5,3.0,5.5])
    cf=np.array([_shape_vector(fem.nodes,float(zi))@yf for zi in z])
    cl=leg.concentration(x,z)
    rel=np.linalg.norm(cl-cf)/np.linalg.norm(cf)
    assert rel < 2.5e-3


def test_08_transient_zero_compensation_typed_api_exactly_reduces_to_legacy_homogeneous_path():
    wind,diff=_constant_profiles(2.1,0.75)
    kwargs=dict(h=7.0,n_modes=18,wind=wind,diffusivity=diff,
                source_height=2.8,emission_rate=1.0,n_quad=80,z_lower=0.1)
    legacy=assemble_transient_legendre_deposition_system(**kwargs,deposition_velocity=0.02)
    typed=assemble_transient_legendre_boundary_system(
        **kwargs,boundary=LinearRobinBoundaryCondition(0.02,0.0,"typed transient")
    )
    for name in ("M","operator","boundary_forcing","G","source_coefficients"):
        np.testing.assert_array_equal(getattr(legacy,name),getattr(typed,name))
    for s in (0.2+0.4j,1.1+0.3j):
        np.testing.assert_allclose(legacy.laplace_coefficients(1.2,s),typed.laplace_coefficients(1.2,s),rtol=0,atol=0)


def test_09_transient_affine_laplace_residual_and_constant_forcing_transform_are_exact():
    boundary=LinearRobinBoundaryCondition(0.028,2.3,"transient affine")
    wind,diff=_constant_profiles(1.7,0.55)
    system=assemble_transient_legendre_boundary_system(
        h=6.0,n_modes=20,wind=wind,diffusivity=diff,source_height=2.0,
        emission_rate=1.0,boundary=boundary,n_quad=90,z_lower=0.0,
    )
    b=lower_values(h=6.0,n_modes=20,z_lower=0.0)
    np.testing.assert_allclose(system.boundary_forcing,0.028*2.3*b,rtol=0,atol=0)
    for s in (0.4+0.7j,1.3+0.2j):
        r=system.laplace_residual(1.4,s)
        scale=max(1.0,np.linalg.norm(system.laplace_system_matrix(s)@system.laplace_coefficients(1.4,s)))
        assert np.linalg.norm(r)/scale < 2e-13
        q1=system.laplace_x_forcing(s)
        q2=solve(system.M.astype(complex),system.boundary_forcing.astype(complex)/s,assume_a="gen")
        np.testing.assert_allclose(q1,q2,rtol=2e-14,atol=2e-14)


def test_10_transient_final_value_recovers_affine_steady_solution_and_scope_guard():
    boundary=LinearResolvedInterfaceFluxLaw(0.021,1.7,"qa036-to-qa037")
    wind,diff=_constant_profiles(2.0,0.65)
    kwargs=dict(h=7.5,n_modes=24,wind=wind,diffusivity=diff,
                source_height=3.0,emission_rate=1.2,n_quad=100,z_lower=0.0)
    steady=assemble_legendre_boundary_system(**kwargs,boundary=boundary)
    transient=assemble_transient_legendre_boundary_system(**kwargs,boundary=boundary)
    s=1e-8
    for x,z in ((0.5,0.3),(1.5,2.0),(4.0,6.0)):
        c_final=s*transient.laplace_concentration(x,z,s)
        c_steady=float(steady.concentration(x,np.array([z]))[0])
        assert c_final.real == pytest.approx(c_steady,rel=2e-7,abs=2e-8)
        assert abs(c_final.imag) < 1e-11

    import gilttpy.solvers.lower_boundary_operator as opmod
    import gilttpy.solvers.steady_2d_deposition_legendre as stmod
    import gilttpy.solvers.transient_2d_deposition_legendre as trmod
    src="\n".join((inspect.getsource(opmod),inspect.getsource(stmod),inspect.getsource(trmod)))
    assert "aerosol_deposition" not in src
    assert "aerosol_size_distribution" not in src
    assert "historical" in inspect.getsource(stmod).lower()
