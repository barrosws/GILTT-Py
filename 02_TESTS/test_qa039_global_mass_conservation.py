from __future__ import annotations

import math
import inspect

import numpy as np
import pytest

from gilttpy.diagnostics.mass_conservation import (
    aggregate_steady_mass_budgets,
    source_rate_from_inlet_condition,
    steady_mass_budget,
    transient_laplace_mass_budget,
    transient_local_laplace_residual,
)
from gilttpy.physics.aerosol_size_distribution import (
    AerosolDiameterDomain,
    AerosolDistributionWeighting,
    LognormalAerosolMode,
)
from gilttpy.physics.particle_physics import AerosolAirState, AerosolParticleProperties, AerosolParticleTransportState
from gilttpy.physics.aerosol_transport_coupling import (
    venkatram_pleim_resolved_interface_flux_law,
    zhang2001_split_resolved_interface_flux_law,
)
from gilttpy.physics.surface_abstraction import ProvenanceRecord
from gilttpy.solvers.lower_boundary_operator import LinearRobinBoundaryCondition
from gilttpy.solvers.settling_2d_legendre import (
    assemble_settling_legendre_system,
    assemble_transient_settling_legendre_system,
)


def _profiles(u=2.0,k=0.7):
    return (
        lambda z: np.full_like(np.asarray(z,dtype=float),u,dtype=float),
        lambda z: np.full_like(np.asarray(z,dtype=float),k,dtype=float),
    )


def _system(*,vg=0.012,kb=0.025,q=1.0,n=26,h=8.0):
    wind,diff=_profiles()
    boundary=LinearRobinBoundaryCondition(kb,0.0,"qa039 total lower flux")
    return assemble_settling_legendre_system(
        h=h,n_modes=n,wind=wind,diffusivity=diff,source_height=0.41*h,
        emission_rate=q,settling_velocity_m_s=vg,boundary=boundary,
        n_quad=120,z_lower=0.0,
    )


def test_01_inlet_condition_represents_exact_source_rate_and_steady_global_budget_closes():
    sys=_system(vg=0.014,kb=0.031,q=1.37,n=30)
    assert source_rate_from_inlet_condition(sys) == pytest.approx(1.37,rel=2e-15,abs=2e-15)
    b=steady_mass_budget(sys,4.5)
    assert b.inlet_advective_flux == pytest.approx(b.source_rate,rel=3e-13,abs=3e-13)
    assert b.relative_residual < 2e-12
    assert b.deposited_fraction + b.outlet_fraction == pytest.approx(1.0,rel=2e-12,abs=2e-12)


def test_02_integrated_global_budget_matches_local_constant_test_identity_over_multiple_fetches():
    sys=_system(vg=0.018,kb=0.034,n=32)
    for x in (0.0,0.2,1.0,3.0,8.0):
        b=steady_mass_budget(sys,x)
        assert b.relative_residual < 3e-12
        if x>0:
            assert sys.flux_derivative_from_weak_constant_test(x) == pytest.approx(
                -sys.lower_boundary_downward_flux(x),rel=5e-11,abs=5e-11
            )
    # Point-source spectral truncation is not positivity preserving arbitrarily
    # close to x=0. Conservation is the gate here, not a hidden clipping rule.
    assert steady_mass_budget(sys,0.2).integrated_lower_deposition < 0.0
    b8=steady_mass_budget(sys,8.0)
    assert b8.integrated_lower_deposition > 0.0
    # Independent x-quadrature checks the augmented-matrix integral used by the
    # budget diagnostic rather than merely reusing its algebra.
    gx,gw=np.polynomial.legendre.leggauss(384)
    xx=4.0*(gx+1.0)
    dep_quad=4.0*float(np.dot(gw,np.asarray([sys.lower_boundary_downward_flux(float(v)) for v in xx])))
    assert dep_quad == pytest.approx(b8.integrated_lower_deposition,rel=2e-10,abs=2e-12)


def test_03_zero_settling_global_budget_is_exactly_conservative_and_has_no_hidden_settling_loss():
    sys=_system(vg=0.0,kb=0.02,n=24)
    b=steady_mass_budget(sys,6.0)
    assert b.relative_residual < 2e-12
    assert b.weak_top_flux == 0.0
    assert b.integrated_lower_deposition > 0.0
    assert b.outlet_advective_flux > 0.0
    _,diff=_profiles()
    strong_top=float(sys.downward_vertical_flux(6.0,np.asarray([sys.h]),diffusivity=diff)[0])
    assert abs(strong_top) < 1e-10


def test_04_strong_settling_budget_closes_and_surface_fraction_increases_downwind():
    weak=_system(vg=0.002,kb=0.012,n=34)
    strong=_system(vg=0.20,kb=0.205,n=34)
    bw=steady_mass_budget(weak,6.0)
    bs=steady_mass_budget(strong,6.0)
    assert bw.relative_residual < 3e-11
    assert bs.relative_residual < 3e-10
    assert bs.deposited_fraction > bw.deposited_fraction
    assert 0.0 <= bs.outlet_fraction <= 1.0+1e-10


def test_05_vp_resolved_interface_family_closes_source_to_surface_without_double_counting():
    vg=0.015
    law=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=vg,unresolved_resistance_s_m=55.0,
        provenance="QA039 VP global budget",
    )
    wind,diff=_profiles(2.1,0.65)
    sys=assemble_settling_legendre_system(
        h=7.0,n_modes=30,wind=wind,diffusivity=diff,source_height=2.9,
        emission_rate=1.0,settling_velocity_m_s=vg,boundary=law,n_quad=130,z_lower=0.0,
    )
    b=steady_mass_budget(sys,7.0)
    assert b.relative_residual < 5e-12
    # The total lower flux is counted once: source partitions only into outlet+surface.
    assert b.source_rate == pytest.approx(b.outlet_advective_flux+b.integrated_lower_deposition,rel=5e-12,abs=5e-12)


def test_06_zhang_local_split_is_globally_conservative_without_claiming_reference_height_invariance():
    vg=0.012
    law=zhang2001_split_resolved_interface_flux_law(
        settling_velocity_m_s=vg,residual_aerodynamic_resistance_s_m=8.0,
        surface_resistance_s_m=42.0,provenance="QA039 Z01 local split budget",
    )
    wind,diff=_profiles(2.0,0.72)
    sys=assemble_settling_legendre_system(
        h=7.5,n_modes=30,wind=wind,diffusivity=diff,source_height=3.0,
        emission_rate=1.0,settling_velocity_m_s=vg,boundary=law,n_quad=130,z_lower=0.0,
    )
    b=steady_mass_budget(sys,6.0)
    assert b.relative_residual < 6e-12
    assert "split" in law.model_family
    # QA039 does not reinterpret the local split as the original reference-height closure.
    import gilttpy.diagnostics.mass_conservation as mod
    src=inspect.getsource(mod)
    assert "aerosol_deposition" not in src
    assert "zhang2001" not in src.lower()


def test_07_positive_sink_drives_source_budget_toward_surface_removal_at_large_fetch():
    sys=_system(vg=0.20,kb=0.205,n=26)
    b_short=steady_mass_budget(sys,10.0)
    b_long=steady_mass_budget(sys,500.0)
    assert b_long.relative_residual < 2e-10
    assert b_long.integrated_lower_deposition > b_short.integrated_lower_deposition
    assert b_long.outlet_advective_flux < b_short.outlet_advective_flux
    assert b_long.deposited_fraction > 0.999


def test_08_transient_laplace_local_and_global_budgets_include_storage_exactly():
    vg=0.014
    law=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=vg,unresolved_resistance_s_m=50.0,provenance="QA039 transient"
    )
    wind,diff=_profiles(2.0,0.68)
    tr=assemble_transient_settling_legendre_system(
        h=7.0,n_modes=24,wind=wind,diffusivity=diff,source_height=2.8,
        emission_rate=1.1,settling_velocity_m_s=vg,boundary=law,n_quad=100,z_lower=0.0,
    )
    for s in (0.25+0.35j,1.2+0.4j,3.0+0.8j):
        for x in (0.2,1.0,3.0):
            r=transient_local_laplace_residual(tr,x,s)
            assert abs(r) < 2e-11
        b=transient_laplace_mass_budget(tr,4.0,s)
        assert b.relative_residual < 2e-11
        assert b.inlet_advective_flux_transform == pytest.approx(b.source_transform,rel=2e-12,abs=2e-12)


def test_09_transient_final_value_global_budget_recovers_steady_and_storage_rate_vanishes():
    vg=0.01
    law=venkatram_pleim_resolved_interface_flux_law(
        settling_velocity_m_s=vg,unresolved_resistance_s_m=60.0,provenance="QA039 final-value budget"
    )
    wind,diff=_profiles(2.0,0.7)
    kwargs=dict(
        h=6.0,n_modes=20,wind=wind,diffusivity=diff,source_height=2.5,
        emission_rate=1.0,settling_velocity_m_s=vg,boundary=law,n_quad=90,z_lower=0.0,
    )
    st=assemble_settling_legendre_system(**kwargs)
    tr=assemble_transient_settling_legendre_system(**kwargs)
    steady=steady_mass_budget(st,3.0)
    s=1e-8
    lap=transient_laplace_mass_budget(tr,3.0,s)
    assert (s*lap.outlet_advective_flux_transform).real == pytest.approx(steady.outlet_advective_flux,rel=3e-7,abs=3e-8)
    assert (s*lap.integrated_lower_deposition_transform).real == pytest.approx(steady.integrated_lower_deposition,rel=3e-7,abs=3e-8)
    assert abs(s*lap.storage_rate_transform) < 3e-8
    assert (s*lap.source_transform).real == pytest.approx(steady.source_rate,rel=3e-12,abs=3e-12)

def test_10_lognormal_mass_weighted_size_aggregation_preserves_global_budget():
    prov=ProvenanceRecord(citation="QA039 synthetic distribution",applicability="structural mass budget only",version="qa039")
    mode=LognormalAerosolMode(
        label="qa039 mode",total_number_concentration_m3=1e8,
        geometric_mean_diameter_m=1.0e-6,geometric_std_dev=1.8,density_kg_m3=1500.0,
        diameter_basis="dry_current_transport",provenance=prov,
    )
    domain=AerosolDiameterDomain(0.1e-6,8e-6,prov)
    retained=mode.retained_fraction(domain=domain,weighting=AerosolDistributionWeighting.MASS)
    assert 0.0 < retained <= 1.0

    # Independent Gauss-Legendre quadrature of the mass-weighted lognormal on
    # the QA035 explicit diameter domain.  Each size class receives a source
    # proportional to its conditional in-domain mass probability.
    qx,qw=np.polynomial.legendre.leggauss(18)
    a=math.log(domain.min_diameter_m); b=math.log(domain.max_diameter_m)
    xs=0.5*(b-a)*qx+0.5*(a+b)
    mu=math.log(mode.geometric_mean_diameter_m)+3.0*math.log(mode.geometric_std_dev)**2
    sig=math.log(mode.geometric_std_dev)
    pdf=np.exp(-0.5*((xs-mu)/sig)**2)/(sig*math.sqrt(2*math.pi))
    weights=0.5*(b-a)*qw*pdf/retained
    weights=weights/np.sum(weights)

    wind,diff=_profiles(2.0,0.7)
    items=[]
    for xlog,w in zip(xs,weights):
        d=math.exp(float(xlog))
        particle=AerosolParticleProperties(
            diameter_m=d,density_kg_m3=mode.density_kg_m3,
            diameter_basis=mode.diameter_basis,provenance="QA039 size node",
        )
        vg=AerosolParticleTransportState(
            particle=particle,air=AerosolAirState(temperature_k=298.15,pressure_pa=101325.0)
        ).settling_velocity_m_s
        law=venkatram_pleim_resolved_interface_flux_law(
            settling_velocity_m_s=vg,unresolved_resistance_s_m=55.0,
            provenance="QA039 size-resolved VP structural budget",
        )
        sys=assemble_settling_legendre_system(
            h=6.5,n_modes=18,wind=wind,diffusivity=diff,source_height=2.7,
            emission_rate=1.0,settling_velocity_m_s=vg,boundary=law,n_quad=80,z_lower=0.0,
        )
        items.append((float(w),steady_mass_budget(sys,4.0)))
    agg=aggregate_steady_mass_budgets(items)
    assert sum(float(w) for w in weights) == pytest.approx(1.0,rel=2e-15,abs=2e-15)
    assert agg.relative_residual < 2e-10
    assert agg.source_rate == pytest.approx(1.0,rel=2e-13,abs=2e-13)
    assert agg.outlet_advective_flux+agg.integrated_lower_deposition == pytest.approx(1.0,rel=2e-10,abs=2e-10)
