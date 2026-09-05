import inspect
import math

import pytest

from gilttpy.physics.aerosol_deposition import (
    AerosolDepositionMeteorology,
    AerosolDepositionModelFamily,
    SETTLING_COUPLING_STATUS,
    StandaloneVenkatramPleimAerosolDeposition,
    StandaloneZhang2001AerosolDeposition,
    VenkatramPleimNonsettlingResistance,
    ZHANG2001_EPSILON0,
    zhang2001_complete_deposition_velocity_m_s,
    zhang2001_surface_resistance_s_m,
)
from gilttpy.physics.deposition import AerosolResistanceSettling
from gilttpy.physics.aerosol_collection import AerosolCollectionState, Zhang2001CollectionEfficiencies
from gilttpy.physics.particle_physics import AerosolAirState, AerosolParticleTransportState
from gilttpy.physics.particle_physics import AerosolParticleProperties
from gilttpy.physics.surface_abstraction import (
    AerosolSurfaceRegime,
    GasPathParameterSet,
    GasSurfaceParameterSet,
    ProvenanceRecord,
    RoughAerosolParameterSet,
    SmoothAerosolParameterSet,
    SurfaceDescriptor,
    SurfacePhysicsBundle,
    SurfaceState,
)


def _prov(label):
    return ProvenanceRecord(
        citation=label,
        applicability="QA034 synthetic structural audit; not calibration",
        version="QA-only",
    )


def _gas_stub():
    return GasSurfaceParameterSet(
        paths=GasPathParameterSet(
            external_surface_resistance_s_m=500.0,
            in_canopy_resistance_s_m=100.0,
            soil_resistance_s_m=300.0,
            provenance=_prov("gas stub"),
        ),
        stomatal=None,
    )


def _surface(*, rough=True, wet=False):
    regime = AerosolSurfaceRegime.VEGETATED_ROUGH if rough else AerosolSurfaceRegime.SMOOTH
    descriptor = SurfaceDescriptor(
        surface_label="QA rough" if rough else "QA smooth",
        land_use_label="arbitrary QA class",
        aerosol_regime=regime,
        provenance=_prov("surface identity"),
        vegetation_type=None,
    )
    state = SurfaceState(descriptor=descriptor, is_wet=wet, provenance=_prov("surface state"))
    aerosol = (
        RoughAerosolParameterSet(
            brownian_exponent=2.0 / 3.0,
            impaction_alpha=1.0,
            collector_radius_m=2.0e-3,
            rebound_activation_diameter_m=5.0e-6,
            provenance=_prov("rough aerosol parameters"),
        )
        if rough
        else SmoothAerosolParameterSet(
            brownian_exponent=0.5,
            rebound_activation_diameter_m=5.0e-6,
            provenance=_prov("smooth aerosol parameters"),
        )
    )
    return SurfacePhysicsBundle(state=state, gas=_gas_stub(), aerosol=aerosol)


def _met(ustar=0.4):
    return AerosolDepositionMeteorology(
        friction_velocity_m_s=ustar,
        temperature_k=298.15,
        pressure_pa=101325.0,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.01,
        provenance=_prov("meteorology"),
        monin_obukhov_length_m=math.inf,
    )


def _particle(dp_um=1.0):
    return AerosolParticleProperties(
        diameter_m=dp_um * 1e-6,
        density_kg_m3=1500.0,
        diameter_basis="current transport diameter",
        provenance="QA034 synthetic particle",
    )


def test_zhang2001_surface_resistance_exact_source_algebra_and_closed_limit():
    rs = zhang2001_surface_resistance_s_m(
        friction_velocity_m_s=0.4,
        total_collection_efficiency=0.2,
        sticking_fraction=0.5,
    )
    assert rs == pytest.approx(1.0 / (3.0 * 0.4 * 0.2 * 0.5), rel=1e-15)
    assert ZHANG2001_EPSILON0 == 3.0
    assert math.isinf(
        zhang2001_surface_resistance_s_m(
            friction_velocity_m_s=0.4,
            total_collection_efficiency=0.0,
            sticking_fraction=1.0,
        )
    )


def test_zhang2001_complete_velocity_exact_additive_settling_identity():
    vd = zhang2001_complete_deposition_velocity_m_s(
        settling_velocity_m_s=0.002,
        aerodynamic_resistance_s_m=20.0,
        surface_resistance_s_m=80.0,
    )
    assert vd == pytest.approx(0.002 + 1.0 / 100.0, rel=1e-15)
    assert zhang2001_complete_deposition_velocity_m_s(
        settling_velocity_m_s=0.002,
        aerodynamic_resistance_s_m=20.0,
        surface_resistance_s_m=math.inf,
    ) == pytest.approx(0.002, rel=0.0, abs=0.0)


def test_complete_rough_provider_matches_independent_qa030_component_assembly():
    particle = _particle(3.0)
    surface = _surface(rough=True)
    met = _met()
    result = StandaloneZhang2001AerosolDeposition(
        particle=particle, surface=surface, meteorology=met
    ).result()

    # Independent assembly from pre-QA034 component providers.
    transport = AerosolParticleTransportState(
        particle=particle, air=AerosolAirState(met.temperature_k, met.pressure_pa)
    )
    p = surface.rough_collection_parameters()
    st = transport.stokes_number(
        friction_velocity_m_s=met.friction_velocity_m_s,
        surface_regime="vegetated",
        collector_radius_m=p.collector_radius_m,
    )
    eff = Zhang2001CollectionEfficiencies(
        state=AerosolCollectionState(transport.schmidt_number, st, particle.diameter_m),
        surface=p,
    ).efficiencies()
    r1 = surface.rebound_provider().sticking_fraction(
        stokes_number=st, particle_diameter_m=particle.diameter_m
    )
    manual_rs = 1.0 / (3.0 * met.friction_velocity_m_s * eff.total * r1)
    manual_ra = met.aerodynamic_provider().resistance_s_m()
    manual_vd = transport.settling_velocity_m_s + 1.0 / (manual_ra + manual_rs)

    assert result.model_family is AerosolDepositionModelFamily.ZHANG2001_SLINN
    assert result.surface_regime is AerosolSurfaceRegime.VEGETATED_ROUGH
    assert result.stokes_number == pytest.approx(st, rel=1e-15)
    assert result.rs_s_m == pytest.approx(manual_rs, rel=1e-15)
    assert result.deposition_velocity_m_s == pytest.approx(manual_vd, rel=1e-15)


def test_complete_smooth_provider_uses_no_collector_interception():
    result = StandaloneZhang2001AerosolDeposition(
        particle=_particle(10.0), surface=_surface(rough=False), meteorology=_met()
    ).result()
    assert result.surface_regime is AerosolSurfaceRegime.SMOOTH
    assert result.interception_efficiency == 0.0
    assert result.impaction_efficiency >= 0.0
    assert result.brownian_efficiency > 0.0
    assert math.isfinite(result.deposition_velocity_m_s)


def test_wet_surface_coarse_particle_cannot_reduce_z01_deposition():
    dry = StandaloneZhang2001AerosolDeposition(
        particle=_particle(10.0), surface=_surface(rough=True, wet=False), meteorology=_met()
    ).result()
    wet = StandaloneZhang2001AerosolDeposition(
        particle=_particle(10.0), surface=_surface(rough=True, wet=True), meteorology=_met()
    ).result()
    assert wet.sticking_fraction == 1.0
    assert wet.sticking_fraction > dry.sticking_fraction
    assert wet.rs_s_m < dry.rs_s_m
    assert wet.deposition_velocity_m_s > dry.deposition_velocity_m_s


def test_venkatram_pleim_wrapper_exactly_reuses_verified_mass_consistent_kernel():
    resistance = VenkatramPleimNonsettlingResistance(
        resistance_s_m=120.0,
        surface_label="QA explicit VP surface",
        provenance=_prov("VP resistance source"),
    )
    model = StandaloneVenkatramPleimAerosolDeposition(
        particle=_particle(3.0), resistance=resistance, meteorology=_met()
    )
    result = model.result()
    expected = AerosolResistanceSettling(
        result.ra_s_m, result.rb_s_m, result.settling_velocity_m_s
    ).deposition_velocity()
    assert result.model_family is AerosolDepositionModelFamily.VENKATRAM_PLEIM_1999
    assert result.deposition_velocity_m_s == pytest.approx(expected, rel=0.0, abs=0.0)


def test_complete_families_remain_numerically_distinct_under_same_resistance_number():
    z01 = StandaloneZhang2001AerosolDeposition(
        particle=_particle(3.0), surface=_surface(rough=True), meteorology=_met()
    ).result()
    vp = StandaloneVenkatramPleimAerosolDeposition(
        particle=_particle(3.0),
        resistance=VenkatramPleimNonsettlingResistance(
            resistance_s_m=z01.rs_s_m,
            surface_label="QA algebraic same-resistance comparison only",
            provenance=_prov("same numerical resistance diagnostic"),
        ),
        meteorology=_met(),
    ).result()
    relative_gap = abs(z01.deposition_velocity_m_s - vp.deposition_velocity_m_s) / z01.deposition_velocity_m_s
    assert relative_gap > 0.05
    assert z01.model_family is not vp.model_family


def test_vp_resistance_is_explicit_and_not_auto_constructed_from_z01_surface_bundle():
    signature = inspect.signature(StandaloneVenkatramPleimAerosolDeposition)
    assert "resistance" in signature.parameters
    assert "surface" not in signature.parameters
    source = inspect.getsource(StandaloneVenkatramPleimAerosolDeposition)
    assert "SurfacePhysicsBundle" not in source
    assert "Zhang2001" not in source


def test_both_complete_closures_expose_settling_partition_hold_and_no_giltt_import():
    z01 = StandaloneZhang2001AerosolDeposition(
        particle=_particle(1.0), surface=_surface(rough=True), meteorology=_met()
    ).result()
    vp = StandaloneVenkatramPleimAerosolDeposition(
        particle=_particle(1.0),
        resistance=VenkatramPleimNonsettlingResistance(100.0, "QA VP", _prov("VP")),
        meteorology=_met(),
    ).result()
    assert z01.settling_coupling_status == SETTLING_COUPLING_STATUS
    assert vp.settling_coupling_status == SETTLING_COUPLING_STATUS
    import gilttpy.physics.aerosol_deposition as ad
    source = inspect.getsource(ad)
    assert "ResolvedLowerInterface" not in source
    assert "steady_2d" not in source
    assert "transient_2d" not in source


def test_reference_size_sweep_is_finite_nonnegative_and_preserves_family_labels():
    z_values = []
    vp_values = []
    for dp_um in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 5.01, 10.0, 20.0):
        z = StandaloneZhang2001AerosolDeposition(
            particle=_particle(dp_um), surface=_surface(rough=True), meteorology=_met()
        ).result()
        v = StandaloneVenkatramPleimAerosolDeposition(
            particle=_particle(dp_um),
            resistance=VenkatramPleimNonsettlingResistance(
                100.0, "QA VP fixed resistance", _prov("VP fixed-resistance size sweep")
            ),
            meteorology=_met(),
        ).result()
        assert math.isfinite(z.deposition_velocity_m_s) and z.deposition_velocity_m_s >= 0.0
        assert math.isfinite(v.deposition_velocity_m_s) and v.deposition_velocity_m_s >= 0.0
        z_values.append(z.deposition_velocity_m_s)
        vp_values.append(v.deposition_velocity_m_s)
    assert len(set(z_values)) > 1
    assert len(set(vp_values)) > 1
