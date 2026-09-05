import numpy as np
from scipy.integrate import quad

from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.variable_coefficient import (
    ManufacturedVariableCoefficientCase,
    assemble_fem_variable_coefficient_reference,
    project_profile_to_legendre,
    spectral_solution_from_initial_coefficients,
)


def _case(**kwargs):
    return ManufacturedVariableCoefficientCase(**kwargs)


def _spectral(case, n_modes=40, n_quad=320):
    sys = assemble_settling_legendre_system(
        h=case.h,
        n_modes=n_modes,
        wind=case.wind,
        diffusivity=case.diffusivity,
        source_height=0.5*(case.z_lower+case.h),
        emission_rate=1.0,
        settling_velocity_m_s=case.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(
            case.boundary_sink_velocity_m_s, label="QA041 manufactured total lower flux"
        ),
        n_quad=n_quad,
        z_lower=case.z_lower,
    )
    y0 = project_profile_to_legendre(
        lambda z: case.concentration(0.0, z),
        h=case.h, n_modes=n_modes, z_lower=case.z_lower, n_quad=max(512, 8*n_modes),
    )
    return sys, y0


def _fem(case, n_elements):
    return assemble_fem_variable_coefficient_reference(
        z_lower=case.z_lower,
        h=case.h,
        n_elements=n_elements,
        wind=case.wind,
        diffusivity=case.diffusivity,
        settling_velocity_m_s=case.settling_velocity_m_s,
        boundary_sink_velocity_m_s=case.boundary_sink_velocity_m_s,
        initial_profile=lambda z: case.concentration(0.0, z),
        element_quad_order=6,
    )


def _rel_l2(got, ref):
    return np.linalg.norm(got-ref)/np.linalg.norm(ref)


def test_qa041_01_manufactured_pde_identity_is_exact():
    c = _case()
    z = np.linspace(c.z_lower, c.h, 57)
    for x in (0.0, 300.0, 1500.0):
        lhs = c.wind(z)*c.streamwise_derivative(x, z)
        rhs = c.vertical_flux_derivative(x, z)
        np.testing.assert_allclose(lhs, rhs, rtol=3e-15, atol=3e-18)


def test_qa041_02_total_flux_boundary_conditions_are_exact():
    c = _case()
    for x in (0.0, 500.0, 2500.0):
        f = c.downward_vertical_flux(x, np.array([c.z_lower, c.h]))
        lower = c.boundary_sink_velocity_m_s*c.concentration(x, [c.z_lower])[0]
        assert abs(f[0]-lower) < 2e-18
        assert abs(f[1]) < 2e-18


def test_qa041_03_unit_source_global_budget_is_analytic():
    c = _case()
    # Independent numerical integral of u*C at x=0 verifies the chosen amplitude.
    val, _ = quad(lambda zz: float(c.wind([zz])[0]*c.concentration(0.0, [zz])[0]), c.z_lower, c.h,
                  epsabs=1e-13, epsrel=1e-13)
    assert abs(val-1.0) < 5e-13
    for x in (100.0, 1000.0, 5000.0):
        assert abs(c.exact_advective_flux(x)+c.exact_integrated_deposition(x)-1.0) < 2e-15
        assert abs(c.exact_lower_flux(x)-c.streamwise_decay_per_m*c.exact_advective_flux(x)) < 2e-18


def test_qa041_04_coefficients_are_genuinely_variable_and_positive():
    c = _case()
    z = np.linspace(c.z_lower, c.h, 101)
    u = c.wind(z); k = c.diffusivity(z)
    assert np.all(u > 0.0) and np.all(k > 0.0)
    assert np.ptp(u) > 1.0
    assert np.ptp(k) > 1.0
    assert c.settling_velocity_m_s > 0.0
    assert c.boundary_sink_velocity_m_s > 0.0


def test_qa041_05_shifted_legendre_matches_exact_variable_coefficient_solution():
    c = _case()
    sys, y0 = _spectral(c, n_modes=36, n_quad=320)
    z = np.linspace(c.z_lower, c.h, 801)
    worst = 0.0
    for x in (0.0, 250.0, 1000.0, 3000.0):
        got = spectral_solution_from_initial_coefficients(sys, y0, x, z)
        ref = c.concentration(x, z)
        worst = max(worst, _rel_l2(got, ref))
    assert worst < 2e-10


def test_qa041_06_spectral_quadrature_is_stable_for_variable_coefficients():
    c = _case()
    s1, y1 = _spectral(c, n_modes=36, n_quad=180)
    s2, y2 = _spectral(c, n_modes=36, n_quad=420)
    z = np.linspace(c.z_lower, c.h, 501)
    a = spectral_solution_from_initial_coefficients(s1, y1, 2000.0, z)
    b = spectral_solution_from_initial_coefficients(s2, y2, 2000.0, z)
    assert _rel_l2(a, b) < 1e-10


def test_qa041_07_independent_p1_fem_shows_second_order_like_refinement():
    c = _case()
    z = np.linspace(c.z_lower, c.h, 1601)
    ref = c.concentration(1500.0, z)
    errs = []
    for ne in (40, 80, 160):
        got = _fem(c, ne).concentration(1500.0, z)
        errs.append(_rel_l2(got, ref))
    assert errs[1] < errs[0]/3.2
    assert errs[2] < errs[1]/3.2
    assert errs[2] < 2e-5


def test_qa041_08_independent_fem_and_spectral_agree_on_exact_case():
    c = _case()
    sys, y0 = _spectral(c, n_modes=36, n_quad=320)
    fem = _fem(c, 320)
    z = np.linspace(c.z_lower, c.h, 1201)
    for x in (250.0, 1000.0, 3000.0):
        a = spectral_solution_from_initial_coefficients(sys, y0, x, z)
        b = fem.concentration(x, z)
        assert _rel_l2(a, b) < 6e-6


def test_qa041_09_second_variable_coefficient_case_crosschecks_without_retuning():
    c = _case(
        z_lower=5.0, h=85.0,
        diffusivity_lower_m2_s=0.8,
        diffusivity_fractional_increase=1.5,
        vertical_decay_per_m=0.03,
        streamwise_decay_per_m=2.5e-4,
        label="QA041 second prespecified manufactured case",
        provenance="QA041 second target-free coefficient family",
    )
    sys, y0 = _spectral(c, n_modes=42, n_quad=360)
    fem = _fem(c, 360)
    z = np.linspace(c.z_lower, c.h, 1201)
    ref = c.concentration(1800.0, z)
    sp = spectral_solution_from_initial_coefficients(sys, y0, 1800.0, z)
    fe = fem.concentration(1800.0, z)
    assert _rel_l2(sp, ref) < 2e-10
    assert _rel_l2(fe, ref) < 8e-6
    assert _rel_l2(sp, fe) < 8e-6


def test_qa041_10_smooth_source_has_no_near_source_undershoot_at_moderate_order():
    c = _case()
    sys, y0 = _spectral(c, n_modes=16, n_quad=192)
    z = np.linspace(c.z_lower, c.h, 2001)
    for x in (0.0, 25.0, 100.0):
        got = spectral_solution_from_initial_coefficients(sys, y0, x, z)
        ref = c.concentration(x, z)
        assert np.min(ref) > 0.0
        assert np.min(got) > 0.0
        assert _rel_l2(got, ref) < 2e-8
