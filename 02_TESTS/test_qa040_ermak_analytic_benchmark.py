import numpy as np

from gilttpy.diagnostics.mass_conservation import steady_mass_budget
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import assemble_settling_legendre_system
from gilttpy.validation.ermak import (
    ermak_advective_flux_to_infinity,
    ermak_crosswind_integrated_concentration,
    ermak_ground_deposition_flux,
    ermak_integrated_deposition,
    ermak_reflected_gaussian_limit,
)


def _const(value):
    return lambda z: np.full_like(np.asarray(z, dtype=float), float(value), dtype=float)


def _system(*, n_modes=120, h=500.0, u=5.0, k=2.0, hs=50.0, vg=0.02, vdep=0.03):
    return assemble_settling_legendre_system(
        h=h,
        n_modes=n_modes,
        wind=_const(u),
        diffusivity=_const(k),
        source_height=hs,
        emission_rate=1.0,
        settling_velocity_m_s=vg,
        boundary=LinearRobinBoundaryCondition(vdep, label="QA040 Ermak total ground flux"),
        n_quad=max(512, 4*n_modes),
        z_lower=0.0,
    )


def _ermak_params(*, u=5.0, k=2.0, hs=50.0, vg=0.02, vdep=0.03):
    return dict(
        emission_rate=1.0,
        wind_speed=u,
        diffusivity=k,
        source_height=hs,
        settling_velocity=vg,
        deposition_velocity=vdep,
    )


def test_qa040_01_ermak_reduces_exactly_to_reflected_gaussian():
    z = np.linspace(0.0, 180.0, 301)
    p = dict(emission_rate=1.7, wind_speed=4.2, diffusivity=1.6, source_height=37.0)
    a = ermak_crosswind_integrated_concentration(
        x=430.0, z=z, settling_velocity=0.0, deposition_velocity=0.0, **p
    )
    b = ermak_reflected_gaussian_limit(x=430.0, z=z, **p)
    np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-15)


def test_qa040_02_ermak_satisfies_total_ground_flux_boundary():
    p = _ermak_params()
    dz = 1.0e-4
    for x in (100.0, 500.0, 2000.0):
        z = dz*np.arange(5, dtype=float)
        c = ermak_crosswind_integrated_concentration(x=x, z=z, **p)
        # 5-point forward derivative at z=0.
        dc = (-25*c[0] + 48*c[1] - 36*c[2] + 16*c[3] - 3*c[4])/(12*dz)
        recovered = (p["diffusivity"]*dc + p["settling_velocity"]*c[0])/c[0]
        assert abs(recovered - p["deposition_velocity"]) < 1e-8


def test_qa040_03_ermak_independent_global_mass_budget():
    p = _ermak_params()
    for x in (100.0, 500.0, 2000.0, 5000.0):
        out = ermak_advective_flux_to_infinity(x=x, **p)
        dep = ermak_integrated_deposition(x_end=x, **p)
        assert abs(1.0 - out - dep) < 5e-12


def test_qa040_04_giltt_matches_ermak_moderate_settling_profiles():
    p = _ermak_params()
    s = _system()
    z = np.linspace(0.0, 180.0, 721)
    for x, tol in ((100.0, 3e-6), (300.0, 2e-9), (1000.0, 5e-9), (3000.0, 5e-9)):
        ref = ermak_crosswind_integrated_concentration(x=x, z=z, **p)
        got = s.concentration(x, z)
        rel_l2 = np.linalg.norm(got-ref)/np.linalg.norm(ref)
        assert rel_l2 < tol


def test_qa040_05_giltt_matches_ermak_surface_deposition_flux():
    p = _ermak_params()
    s = _system()
    for x, tol in ((100.0, 3e-6), (500.0, 5e-9), (2000.0, 5e-9)):
        ref = ermak_ground_deposition_flux(x=x, **p)
        got = s.lower_boundary_downward_flux(x)
        assert abs(got-ref)/max(abs(ref), 1e-300) < tol


def test_qa040_06_giltt_matches_ermak_strong_settling_case():
    p = _ermak_params(vg=0.08, vdep=0.12)
    s = _system(vg=0.08, vdep=0.12)
    z = np.linspace(0.0, 180.0, 721)
    for x, tol in ((100.0, 3e-6), (300.0, 3e-9), (1000.0, 5e-9)):
        ref = ermak_crosswind_integrated_concentration(x=x, z=z, **p)
        got = s.concentration(x, z)
        rel_l2 = np.linalg.norm(got-ref)/np.linalg.norm(ref)
        assert rel_l2 < tol


def test_qa040_07_finite_top_converges_to_semi_infinite_ermak_reference():
    p = _ermak_params()
    z = np.linspace(0.0, 150.0, 601)
    ref = ermak_crosswind_integrated_concentration(x=5000.0, z=z, **p)
    s200 = _system(h=200.0)
    s300 = _system(h=300.0)
    e200 = np.linalg.norm(s200.concentration(5000.0, z)-ref)/np.linalg.norm(ref)
    e300 = np.linalg.norm(s300.concentration(5000.0, z)-ref)/np.linalg.norm(ref)
    assert e200 > 5e-4  # top reflection is deliberately detectable
    assert e300 < 2e-8  # semi-infinite benchmark recovered after top is remote
    assert e300 < e200/1e4


def test_qa040_08_exact_ermak_exposes_coarse_near_source_spectral_undershoot():
    h = 500.0; u = 5.0; k = 2.0; hs = 50.0; x = 100.0
    z = np.linspace(0.0, h, 4001)
    ref = ermak_reflected_gaussian_limit(
        x=x, z=z, emission_rate=1.0, wind_speed=u, diffusivity=k, source_height=hs
    )
    coarse = _system(n_modes=40, h=h, u=u, k=k, hs=hs, vg=0.0, vdep=0.0)
    got = coarse.concentration(x, z)
    neg_peak_ratio = max(0.0, -float(got.min()))/float(ref.max())
    rel_l2 = np.linalg.norm(got-ref)/np.linalg.norm(ref)
    assert neg_peak_ratio > 1e-2
    assert rel_l2 > 3e-2


def test_qa040_09_modal_refinement_reduces_near_source_undershoot_by_orders_of_magnitude():
    h = 500.0; u = 5.0; k = 2.0; hs = 50.0; x = 100.0
    z = np.linspace(0.0, h, 4001)
    ref = ermak_reflected_gaussian_limit(
        x=x, z=z, emission_rate=1.0, wind_speed=u, diffusivity=k, source_height=hs
    )
    coarse = _system(n_modes=40, h=h, u=u, k=k, hs=hs, vg=0.0, vdep=0.0).concentration(x, z)
    fine = _system(n_modes=120, h=h, u=u, k=k, hs=hs, vg=0.0, vdep=0.0).concentration(x, z)
    neg_coarse = max(0.0, -float(coarse.min()))/float(ref.max())
    neg_fine = max(0.0, -float(fine.min()))/float(ref.max())
    l2_coarse = np.linalg.norm(coarse-ref)/np.linalg.norm(ref)
    l2_fine = np.linalg.norm(fine-ref)/np.linalg.norm(ref)
    assert neg_fine < 2e-6
    assert neg_coarse/neg_fine > 1e4
    assert l2_fine < 5e-6
    assert l2_coarse/l2_fine > 1e4


def test_qa040_10_conservation_and_positivity_are_distinct_numerical_properties():
    h = 500.0; u = 5.0; k = 2.0; hs = 50.0; x = 100.0
    z = np.linspace(0.0, h, 4001)
    coarse = _system(n_modes=40, h=h, u=u, k=k, hs=hs, vg=0.0, vdep=0.0)
    c = coarse.concentration(x, z)
    budget = steady_mass_budget(coarse, x)
    assert float(c.min()) < 0.0
    assert budget.relative_residual < 1e-12
