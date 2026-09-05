import numpy as np

from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.variable_coefficient import (
    ManufacturedVariableCoefficientCase,
    assemble_fem_variable_coefficient_reference,
)
from gilttpy.validation.robustness_campaign import (
    GaussianInletProfile,
    advective_flux_from_coefficients,
    concentration_from_coefficients,
    finite_dimensional_budget,
    positivity_diagnostic,
    project_profile_to_system,
    propagated_coefficients,
)


def _system(case, *, n_modes=48, source_height=None):
    if source_height is None:
        source_height = 0.5*(case.z_lower+case.h)
    return assemble_settling_legendre_system(
        h=case.h,
        n_modes=n_modes,
        wind=case.wind,
        diffusivity=case.diffusivity,
        source_height=source_height,
        emission_rate=1.0,
        settling_velocity_m_s=case.settling_velocity_m_s,
        boundary=LinearRobinBoundaryCondition(case.boundary_sink_velocity_m_s, label="QA042 total lower flux"),
        n_quad=max(320, 8*n_modes),
        z_lower=case.z_lower,
    )


def _rel_l2(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)


def test_qa042_01_scale_aware_settling_identity_guard_accepts_high_gradient_case():
    c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=10.0, label="stress", provenance="QA042")
    s = _system(c, n_modes=120)
    assert np.all(np.isfinite(s.operator))
    assert c.settling_velocity_m_s > 0.3


def test_qa042_02_spectral_exact_accuracy_survives_large_coefficient_gradient_range():
    for beta in (0.2, 2.0, 10.0):
        c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=beta, label="gradient", provenance="QA042")
        s = _system(c, n_modes=48)
        y0 = project_profile_to_system(lambda z: c.concentration(0.0, z), s, n_quad=768)
        z = np.linspace(c.z_lower, c.h, 1001)
        worst = 0.0
        for x in (0.0, 250.0, 1000.0, 3000.0):
            got = concentration_from_coefficients(s, propagated_coefficients(s, y0, x), z)
            worst = max(worst, _rel_l2(got, c.concentration(x, z)))
        assert worst < 3e-10


def test_qa042_03_hard_gradient_independent_fem_retains_second_order_refinement():
    c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=10.0, label="hard FEM", provenance="QA042")
    z = np.linspace(c.z_lower, c.h, 1201)
    ref = c.concentration(1500.0, z)
    errs = []
    for ne in (40, 80, 160, 320):
        f = assemble_fem_variable_coefficient_reference(
            z_lower=c.z_lower, h=c.h, n_elements=ne, wind=c.wind, diffusivity=c.diffusivity,
            settling_velocity_m_s=c.settling_velocity_m_s,
            boundary_sink_velocity_m_s=c.boundary_sink_velocity_m_s,
            initial_profile=lambda zz: c.concentration(0.0, zz), element_quad_order=6,
        )
        errs.append(_rel_l2(f.concentration(1500.0, z), ref))
    orders = [np.log(errs[i]/errs[i+1])/np.log(2.0) for i in range(3)]
    assert min(orders) > 1.98
    assert errs[-1] < 5e-6


def test_qa042_04_hard_gradient_spectral_and_fem_cross_discretization_agree():
    c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=10.0, label="hard cross", provenance="QA042")
    s = _system(c, n_modes=48)
    y0 = project_profile_to_system(lambda z: c.concentration(0.0, z), s, n_quad=768)
    f = assemble_fem_variable_coefficient_reference(
        z_lower=c.z_lower, h=c.h, n_elements=320, wind=c.wind, diffusivity=c.diffusivity,
        settling_velocity_m_s=c.settling_velocity_m_s,
        boundary_sink_velocity_m_s=c.boundary_sink_velocity_m_s,
        initial_profile=lambda z: c.concentration(0.0, z), element_quad_order=6,
    )
    z = np.linspace(c.z_lower, c.h, 1001)
    sp = concentration_from_coefficients(s, propagated_coefficients(s, y0, 1500.0), z)
    assert _rel_l2(f.concentration(1500.0, z), sp) < 5e-6


def test_qa042_05_explicit_lower_interface_translation_is_numerically_invariant():
    curves = []
    eta = np.linspace(0.0, 1.0, 801)
    for zl in (0.0, 30.0, 100.0):
        c = ManufacturedVariableCoefficientCase(z_lower=zl, h=zl+100.0, diffusivity_fractional_increase=5.0,
                                                label="translated", provenance="QA042")
        s = _system(c, n_modes=48)
        y0 = project_profile_to_system(lambda z: c.concentration(0.0, z), s, n_quad=768)
        z = zl + 100.0*eta
        curves.append(concentration_from_coefficients(s, propagated_coefficients(s, y0, 1500.0), z))
    assert _rel_l2(curves[1], curves[0]) < 2e-11
    assert _rel_l2(curves[2], curves[0]) < 2e-11


def test_qa042_06_point_source_conservation_survives_interface_proximity():
    c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=5.0, label="point", provenance="QA042")
    for d in (0.5, 2.0, 50.0):
        s = _system(c, n_modes=40, source_height=c.z_lower+d)
        for x in (0.1, 10.0, 100.0):
            assert finite_dimensional_budget(s, s.y0, x).relative_residual < 1e-10


def test_qa042_07_point_source_undershoot_is_raw_and_decays_with_distance_and_refinement():
    c = ManufacturedVariableCoefficientCase(diffusivity_fractional_increase=5.0, label="point", provenance="QA042")
    z = np.linspace(c.z_lower, c.h, 6001)
    s40 = _system(c, n_modes=40, source_height=c.z_lower+0.5)
    s120 = _system(c, n_modes=120, source_height=c.z_lower+0.5)
    near40 = positivity_diagnostic(s40.concentration(0.1, z)).negative_peak_ratio
    near120 = positivity_diagnostic(s120.concentration(0.1, z)).negative_peak_ratio
    far40 = positivity_diagnostic(s40.concentration(100.0, z)).negative_peak_ratio
    assert near40 > 0.2
    assert near120 < near40/5.0
    assert far40 < 1e-6


def test_qa042_08_gaussian_source_requires_provenance_and_normalizes_advective_flux():
    c = ManufacturedVariableCoefficientCase(label="gaussian", provenance="QA042")
    g = GaussianInletProfile(c.z_lower, c.h, c.z_lower+5.0, 2.0, label="g2", provenance="QA042 explicit smoothing")
    p = g.profile(c.wind, n_quad=1024)
    s = _system(c, n_modes=80, source_height=g.center_m)
    y0 = project_profile_to_system(p, s, n_quad=1024)
    assert abs(advective_flux_from_coefficients(s, y0)-1.0) < 2e-12
    try:
        GaussianInletProfile(c.z_lower, c.h, c.z_lower+5.0, 2.0, label="", provenance="")
    except ValueError:
        pass
    else:
        raise AssertionError("missing source provenance must be rejected")


def test_qa042_09_gaussian_width_stress_agrees_with_independent_fem():
    c = ManufacturedVariableCoefficientCase(label="gaussian cross", provenance="QA042")
    z = np.linspace(c.z_lower, c.h, 801)
    for sigma in (2.0, 5.0, 10.0):
        g = GaussianInletProfile(c.z_lower, c.h, c.z_lower+5.0, sigma, label=f"g{sigma}", provenance="QA042")
        p = g.profile(c.wind, n_quad=1024)
        s = _system(c, n_modes=80, source_height=g.center_m)
        y0 = project_profile_to_system(p, s, n_quad=1024)
        f = assemble_fem_variable_coefficient_reference(
            z_lower=c.z_lower, h=c.h, n_elements=240, wind=c.wind, diffusivity=c.diffusivity,
            settling_velocity_m_s=c.settling_velocity_m_s,
            boundary_sink_velocity_m_s=c.boundary_sink_velocity_m_s,
            initial_profile=p, element_quad_order=6,
        )
        sp = concentration_from_coefficients(s, propagated_coefficients(s, y0, 300.0), z)
        assert _rel_l2(f.concentration(300.0, z), sp) < 3e-4


def test_qa042_10_smooth_source_separates_positivity_from_conservation_without_clipping():
    c = ManufacturedVariableCoefficientCase(label="smooth positivity", provenance="QA042")
    g = GaussianInletProfile(c.z_lower, c.h, c.z_lower+2.0, 2.0, label="smooth", provenance="QA042")
    p = g.profile(c.wind, n_quad=1024)
    s = _system(c, n_modes=80, source_height=g.center_m)
    y0 = project_profile_to_system(p, s, n_quad=1024)
    z = np.linspace(c.z_lower, c.h, 2001)
    for x in (0.0, 50.0, 300.0, 1000.0):
        y = propagated_coefficients(s, y0, x)
        diag = positivity_diagnostic(concentration_from_coefficients(s, y, z))
        assert diag.negative_peak_ratio < 1e-8
        assert diag.negative_l1_fraction < 1e-8
        assert finite_dimensional_budget(s, y0, x).relative_residual < 5e-9
