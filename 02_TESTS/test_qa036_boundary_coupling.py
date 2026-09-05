import inspect
import math

import numpy as np
import pytest

from gilttpy.physics.aerodynamic import most_aerodynamic_resistance
from gilttpy.physics.bidirectional_exchange import CanopyNodeExchange, CompensationPath
from gilttpy.physics.boundary_coupling import (
    AEROSOL_BOUNDARY_COUPLING_STATUS,
    AerodynamicResistancePartition,
    LinearResolvedInterfaceFluxLaw,
    ResolvedInterfaceBidirectionalGasBoundary,
    ResolvedInterfaceUnidirectionalGasBoundary,
    most_scalar_transfer_resistance_between_heights,
    parallel_surface_equivalent,
)
from gilttpy.physics.gas_deposition import (
    GasDepositionMeteorology,
    GasExchangeAssumption,
    GasSpeciesDepositionProperties,
)
from gilttpy.physics.stomatal import StomatalEnvironment, StomatalPhysiologyParameters
from gilttpy.physics.surface_abstraction import (
    AerosolSurfaceRegime,
    GasPathParameterSet,
    GasSurfaceParameterSet,
    ProvenanceRecord,
    RoughAerosolParameterSet,
    StomatalParameterSet,
    SurfaceDescriptor,
    SurfacePhysicsBundle,
    SurfaceState,
)
from gilttpy.basis.shifted_legendre import lower_values
from gilttpy.solvers.steady_2d_deposition_legendre import assemble_legendre_deposition_system
from gilttpy.solvers.steady_2d_deposition_fem import assemble_fem_deposition_system


def _prov(label):
    return ProvenanceRecord(label, "QA036 target-free synthetic/reference audit", "QA036-v1")


def _met(*, L=math.inf):
    return GasDepositionMeteorology(
        friction_velocity_m_s=0.4, temperature_k=298.15, pressure_pa=101325.0,
        reference_height_m=10.0, scalar_roughness_length_m=0.01,
        displacement_height_m=0.0, monin_obukhov_length_m=L,
        provenance=_prov("QA036 meteorology"),
    )


def _env():
    return StomatalEnvironment(
        ppfd_umol_m2_s=800.0, temperature_c=25.0, vapor_pressure_deficit_kpa=1.0,
        soil_water_content_m3_m3=0.25, day_of_year=180.0, leaf_area_index=3.0,
    )


def _surface(*, closed=False):
    descriptor = SurfaceDescriptor(
        surface_label="QA036 reference surface", land_use_label="QA custom vegetation",
        aerosol_regime=AerosolSurfaceRegime.VEGETATED_ROUGH,
        provenance=_prov("QA036 surface identity"), vegetation_type=None if closed else "QA grass",
    )
    state = SurfaceState(descriptor, False, _prov("QA036 dry state"))
    r = math.inf if closed else 500.0
    paths = GasPathParameterSet(
        external_surface_resistance_s_m=r,
        in_canopy_resistance_s_m=math.inf if closed else 100.0,
        soil_resistance_s_m=math.inf if closed else 300.0,
        provenance=_prov("QA036 gas path resistances"),
    )
    pars = StomatalPhysiologyParameters(
        vegetation_type="QA grass", maximum_leaf_conductance_m_s=0.005,
        minimum_fraction=0.1, light_response_per_umol=0.003,
        temperature_minimum_c=0.0, temperature_optimum_c=25.0, temperature_maximum_c=40.0,
        vpd_full_open_kpa=0.5, vpd_minimum_open_kpa=3.0,
        soil_wilting_point_m3_m3=0.10, soil_field_capacity_m3_m3=0.30,
    )
    gas = GasSurfaceParameterSet(
        paths=paths,
        stomatal=None if closed else StomatalParameterSet(pars, _prov("QA036 stomata")),
    )
    aerosol = RoughAerosolParameterSet(0.5, 1.0, 2e-3, 5e-6, _prov("QA036 aerosol placeholder"))
    return SurfacePhysicsBundle(state, gas, aerosol)


def _species():
    return GasSpeciesDepositionProperties.from_massman_reference(
        "O3", temperature_k=298.15, pressure_pa=101325.0,
        exchange_assumption=GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL,
        provenance=_prov("QA036 O3 property"), stomatal_reference_species="O3",
    )


def _part(*, L=math.inf, interface=0.03, displacement=0.0):
    return AerodynamicResistancePartition(
        friction_velocity_m_s=0.4, reference_height_m=10.0,
        interface_height_m=interface, scalar_roughness_length_m=0.01,
        displacement_height_m=displacement, monin_obukhov_length_m=L,
    )


def test_most_segment_neutral_formula_and_zero_thickness_limit():
    r = most_scalar_transfer_resistance_between_heights(
        friction_velocity_m_s=0.4, upper_height_m=10.0, lower_height_m=0.01,
    )
    assert r == pytest.approx(math.log(1000.0)/(0.4*0.4), rel=1e-15)
    assert most_scalar_transfer_resistance_between_heights(
        friction_velocity_m_s=0.4, upper_height_m=0.03, lower_height_m=0.03,
    ) == 0.0


def test_most_endpoint_additivity_is_exact_across_neutral_stable_unstable_and_displacement():
    cases = [(math.inf, 0.0, 0.03, 0.01), (100.0, 0.0, 0.03, 0.01),
             (-100.0, 0.0, 0.03, 0.01), (80.0, 0.2, 0.35, 0.21)]
    for L, d, split, z0abs in cases:
        full = most_scalar_transfer_resistance_between_heights(
            friction_velocity_m_s=0.4, upper_height_m=10.0, lower_height_m=z0abs,
            displacement_height_m=d, monin_obukhov_length_m=L,
        )
        upper = most_scalar_transfer_resistance_between_heights(
            friction_velocity_m_s=0.4, upper_height_m=10.0, lower_height_m=split,
            displacement_height_m=d, monin_obukhov_length_m=L,
        )
        lower = most_scalar_transfer_resistance_between_heights(
            friction_velocity_m_s=0.4, upper_height_m=split, lower_height_m=z0abs,
            displacement_height_m=d, monin_obukhov_length_m=L,
        )
        assert full == pytest.approx(upper+lower, rel=3e-15, abs=3e-14)


def test_partition_matches_qa030a_full_ra_and_rejects_interface_below_scalar_endpoint():
    for L in (math.inf, 100.0, -100.0):
        p = _part(L=L)
        qa030a = most_aerodynamic_resistance(
            friction_velocity_m_s=0.4, reference_height_m=10.0,
            scalar_roughness_length_m=0.01, monin_obukhov_length_m=L,
        )
        assert p.full_reference_to_surface_s_m == pytest.approx(qa030a, rel=2e-15)
        assert p.additivity_residual_s_m == pytest.approx(0.0, abs=5e-14)
        assert p.residual_subinterface_s_m < p.full_reference_to_surface_s_m
    with pytest.raises(ValueError, match="at or above"):
        _part(interface=0.005)


def test_unidirectional_qa032_chain_maps_to_residual_interface_and_reconstructs_reference_exactly():
    calc = ResolvedInterfaceUnidirectionalGasBoundary(
        _species(), _surface(), _met(), interface_height_m=0.03, stomatal_environment=_env()
    ).result()
    assert calc.residual_ra_s_m == pytest.approx(math.log(3.0)/(0.4*0.4), rel=1e-15)
    assert calc.interface_sink_velocity_m_s > calc.reference_deposition_velocity_m_s
    assert calc.reconstructed_reference_velocity_m_s() == pytest.approx(
        calc.reference_deposition_velocity_m_s, rel=2e-15
    )
    assert calc.partition_additivity_residual_s_m == pytest.approx(0.0, abs=5e-14)
    assert calc.interface_law.downward_flux(10.0) == pytest.approx(10.0*calc.interface_sink_velocity_m_s)


def test_naive_full_ra_boundary_is_demonstrably_double_counted_and_closed_surface_stays_zero():
    p = _part(); rb=15.0; rc=80.0
    correct_ref = 1.0/(p.full_reference_to_surface_s_m+rb+rc)
    reconstructed = 1.0/(p.upper_most_segment_s_m+p.residual_subinterface_s_m+rb+rc)
    naive_double = 1.0/(p.upper_most_segment_s_m+p.full_reference_to_surface_s_m+rb+rc)
    assert reconstructed == pytest.approx(correct_ref, rel=2e-15)
    assert naive_double < correct_ref
    assert naive_double/correct_ref - 1.0 == pytest.approx(-0.20805, abs=5e-4)
    closed = ResolvedInterfaceUnidirectionalGasBoundary(
        _species(), _surface(closed=True), _met(), 0.03, None
    ).result()
    assert closed.rc_s_m == math.inf
    assert closed.interface_sink_velocity_m_s == 0.0


def test_bidirectional_affine_law_matches_full_canopy_node_in_deposition_equilibrium_emission_regimes():
    p=_part(); rb=12.0
    paths=(CompensationPath("stom",80.0,4.0,_prov("stom")),
           CompensationPath("external",200.0,8.0,_prov("external")))
    b=ResolvedInterfaceBidirectionalGasBoundary(p,rb,paths)
    law=b.interface_law()
    rc,ceq=parallel_surface_equivalent(paths)
    assert b.equivalent_surface_resistance_s_m == pytest.approx(rc, rel=1e-15)
    assert law.equilibrium_concentration == pytest.approx(ceq, rel=1e-15)
    for c_ref in (2.0, ceq, 12.0):
        direct=CanopyNodeExchange(c_ref,p.full_reference_to_surface_s_m,rb,paths).solve()[1]
        # direct is upward-positive, reconstructed is downward-positive
        assert b.reconstructed_reference_downward_flux(c_ref) == pytest.approx(-direct, rel=2e-14, abs=1e-14)


def test_zero_compensation_bidirectional_reduces_exactly_to_unidirectional_parallel_resistance():
    p=_part(); rb=12.0
    paths=(CompensationPath("a",100.0,0.0,_prov("a")),
           CompensationPath("b",300.0,0.0,_prov("b")))
    rc,ceq=parallel_surface_equivalent(paths)
    assert rc == pytest.approx(75.0, rel=1e-15); assert ceq == 0.0
    law=ResolvedInterfaceBidirectionalGasBoundary(p,rb,paths).interface_law()
    expected=1.0/(p.residual_subinterface_s_m+rb+75.0)
    assert law.sink_velocity_m_s == pytest.approx(expected, rel=1e-15)
    assert law.equilibrium_concentration == 0.0


def test_affine_weak_terms_are_exact_boundary_flux_vector_with_correct_sign():
    law=LinearResolvedInterfaceFluxLaw(0.02,5.0,"QA affine Robin")
    b=np.array([1.0,-0.3,0.2])
    y=np.array([7.0,2.0,-1.0])
    B,f=law.weak_terms(b)
    c=float(b@y); j=law.downward_flux(c)
    assert np.allclose(B@y-f,b*j,rtol=0.0,atol=1e-15)
    assert j>0.0
    assert law.downward_flux(5.0)==pytest.approx(0.0,abs=0.0)
    assert law.downward_flux(2.0)<0.0


def test_existing_legendre_and_independent_fem_weak_solvers_obey_interface_mass_balance_for_sink_law():
    p=_part(); law=LinearResolvedInterfaceFluxLaw(1.0/(p.residual_subinterface_s_m+25.0+100.0))
    wind=lambda z: np.ones_like(np.asarray(z,dtype=float))*2.0
    kz=lambda z: np.ones_like(np.asarray(z,dtype=float))*0.5
    leg=assemble_legendre_deposition_system(
        z_lower=0.03,h=20.0,n_modes=24,wind=wind,diffusivity=kz,
        source_height=5.0,emission_rate=1.0,deposition_velocity=law.sink_velocity_m_s,n_quad=96,
    )
    fem=assemble_fem_deposition_system(
        z_lower=0.03,h=20.0,n_elements=120,wind=wind,diffusivity=kz,
        source_height=5.0,emission_rate=1.0,deposition_velocity=law.sink_velocity_m_s,
        grading_power=1.5,element_quad_order=4,
    )
    # At x=0 a finite spectral representation of the point source can undershoot
    # locally.  The weak mass identity is algebraic and remains valid for the
    # signed finite-dimensional trace, so do not route this diagnostic through
    # the physical nonnegative-concentration validator.
    for x in (0.0, 5.0, 20.0):
        c_leg = leg.lower_boundary_concentration(x)
        c_fem = fem.lower_boundary_concentration(x)
        assert leg.flux_derivative_from_weak_constant_test(x) == pytest.approx(
            -law.sink_velocity_m_s * c_leg, rel=3e-12, abs=3e-13)
        assert fem.flux_derivative_from_constant_test(x) == pytest.approx(
            -law.sink_velocity_m_s * c_fem, rel=3e-12, abs=3e-13)


def test_qa036_does_not_activate_aerosol_or_import_transport_operator_and_preserves_qa037_qa038_holds():
    import gilttpy.physics.boundary_coupling as bc
    source=inspect.getsource(bc)
    assert "aerosol_deposition" not in source
    assert "particle_physics" not in source
    assert "steady_2d" not in source
    assert "transient_2d" not in source
    assert "QA038_QA039" in AEROSOL_BOUNDARY_COUPLING_STATUS
    with pytest.raises(ValueError):
        LinearResolvedInterfaceFluxLaw(-1.0)
