import inspect
import math

import pytest

from gilttpy.physics.aerosol_collection import Zhang2001SurfaceCollectionParameters
from gilttpy.physics.stomatal import (
    JarvisEmbersonBulkStomatalResistance,
    StomatalEnvironment,
    StomatalPhysiologyParameters,
)
from gilttpy.physics.surface_abstraction import (
    AerosolSurfaceRegime,
    GasPathParameterSet,
    GasSurfaceParameterSet,
    ProvenanceRecord,
    RoughAerosolParameterSet,
    SmoothAerosolParameterSet,
    StomatalParameterSet,
    SurfaceDescriptor,
    SurfacePhysicsBundle,
    SurfaceState,
)
from gilttpy.physics.surface_resistance import DEPACCanopyResistance


def _prov(label="QA031 synthetic source"):
    return ProvenanceRecord(
        citation=label,
        version="QA-only",
        applicability="synthetic structural audit; not a land-use calibration",
    )


def _stomatal_params(vegetation="QA grass"):
    return StomatalPhysiologyParameters(
        vegetation_type=vegetation,
        maximum_leaf_conductance_m_s=0.005,
        minimum_fraction=0.1,
        light_response_per_umol=0.003,
        temperature_minimum_c=0.0,
        temperature_optimum_c=25.0,
        temperature_maximum_c=40.0,
        vpd_full_open_kpa=0.5,
        vpd_minimum_open_kpa=3.0,
        soil_wilting_point_m3_m3=0.10,
        soil_field_capacity_m3_m3=0.30,
    )


def _env():
    return StomatalEnvironment(
        ppfd_umol_m2_s=800.0,
        temperature_c=25.0,
        vapor_pressure_deficit_kpa=1.0,
        soil_water_content_m3_m3=0.25,
        day_of_year=180.0,
        leaf_area_index=3.0,
    )


def _rough_bundle(*, land_use="custom rough", wet=False):
    descriptor = SurfaceDescriptor(
        surface_label="QA rough surface",
        land_use_label=land_use,
        aerosol_regime=AerosolSurfaceRegime.VEGETATED_ROUGH,
        vegetation_type="QA grass",
        provenance=_prov("surface map source"),
    )
    state = SurfaceState(descriptor=descriptor, is_wet=wet, provenance=_prov("state source"))
    gas = GasSurfaceParameterSet(
        paths=GasPathParameterSet(
            external_surface_resistance_s_m=500.0,
            in_canopy_resistance_s_m=100.0,
            soil_resistance_s_m=300.0,
            provenance=_prov("gas path source"),
        ),
        stomatal=StomatalParameterSet(_stomatal_params(), _prov("stomatal source")),
    )
    aerosol = RoughAerosolParameterSet(
        brownian_exponent=2.0 / 3.0,
        impaction_alpha=1.0,
        collector_radius_m=2.0e-3,
        rebound_activation_diameter_m=5.0e-6,
        provenance=_prov("aerosol source"),
    )
    return SurfacePhysicsBundle(state=state, gas=gas, aerosol=aerosol)


def test_provenance_is_mandatory_and_structured():
    with pytest.raises(ValueError):
        ProvenanceRecord("", "scope", "v1")
    with pytest.raises(ValueError):
        ProvenanceRecord("source", "", "v1")
    p = _prov()
    assert "QA031 synthetic source" in p.compact_label
    assert "QA-only" in p.compact_label


def test_surface_identity_does_not_trigger_numeric_land_use_lookup():
    a = _rough_bundle(land_use="arbitrary class A")
    b = _rough_bundle(land_use="completely different class B")
    pa = a.rough_collection_parameters()
    pb = b.rough_collection_parameters()
    assert pa.brownian_exponent == pb.brownian_exponent
    assert pa.impaction_alpha == pb.impaction_alpha
    assert pa.collector_radius_m == pb.collector_radius_m
    assert a.gas.canopy_provider(environment=_env(), species_to_reference_diffusivity_ratio=1.0).resistance_s_m() == pytest.approx(
        b.gas.canopy_provider(environment=_env(), species_to_reference_diffusivity_ratio=1.0).resistance_s_m(), rel=0.0, abs=0.0
    )


def test_rough_surface_generates_exact_existing_zhang_parameter_contract():
    bundle = _rough_bundle()
    p = bundle.rough_collection_parameters()
    assert isinstance(p, Zhang2001SurfaceCollectionParameters)
    assert p.surface_label == "QA rough surface"
    assert p.brownian_exponent == pytest.approx(2.0 / 3.0)
    assert p.impaction_alpha == 1.0
    assert p.collector_radius_m == 2.0e-3
    assert "aerosol source" in p.provenance


def test_smooth_and_rough_aerosol_topologies_cannot_be_cross_wired():
    descriptor = SurfaceDescriptor(
        "smooth QA", "paved", AerosolSurfaceRegime.SMOOTH, _prov("surface"), None
    )
    state = SurfaceState(descriptor, False, _prov("state"))
    gas = GasSurfaceParameterSet(
        GasPathParameterSet(100.0, 100.0, 100.0, _prov("gas")), None
    )
    rough = RoughAerosolParameterSet(0.5, 1.0, 1e-3, 5e-6, _prov("rough"))
    with pytest.raises(ValueError):
        SurfacePhysicsBundle(state, gas, rough)

    smooth = SmoothAerosolParameterSet(0.5, 5e-6, _prov("smooth"))
    bundle = SurfacePhysicsBundle(state, gas, smooth)
    with pytest.raises(ValueError):
        bundle.rough_collection_parameters()


def test_surface_wetness_round_trips_exactly_to_existing_aerosol_state():
    dry = _rough_bundle(wet=False)
    wet = _rough_bundle(wet=True)
    assert dry.aerosol_surface_state().is_wet is False
    assert wet.aerosol_surface_state().is_wet is True
    assert dry.aerosol_surface_state().surface_label == "QA rough surface"
    assert "state source" in dry.aerosol_surface_state().provenance


def test_stomatal_and_canopy_provider_chain_matches_direct_qa030d_qa030c_use():
    bundle = _rough_bundle()
    env = _env()
    ratio = 0.92
    with pytest.raises(ValueError):
        bundle.gas.canopy_provider(environment=env)
    resolved = bundle.gas.canopy_provider(
        environment=env, species_to_reference_diffusivity_ratio=ratio
    )
    direct_rstom = JarvisEmbersonBulkStomatalResistance(
        env, bundle.gas.stomatal.parameters, ratio
    ).resistance_s_m()
    direct = DEPACCanopyResistance(direct_rstom, 500.0, 100.0, 300.0)
    assert resolved.resistance_s_m() == pytest.approx(direct.resistance_s_m(), rel=1e-15)


def test_nonstomatal_surface_closes_stomatal_path_without_fake_finite_cap():
    gas = GasSurfaceParameterSet(
        paths=GasPathParameterSet(200.0, 50.0, 250.0, _prov("nonstomatal gas")),
        stomatal=None,
    )
    assert gas.stomatal_resistance_s_m(environment=None) == math.inf
    rc = gas.canopy_provider(environment=None).resistance_s_m()
    expected = DEPACCanopyResistance(math.inf, 200.0, 50.0, 250.0).resistance_s_m()
    assert rc == pytest.approx(expected, rel=0.0, abs=0.0)
    with pytest.raises(ValueError):
        gas.stomatal_resistance_s_m(environment=_env())


def test_vegetation_identity_must_match_stomatal_parameter_identity():
    descriptor = SurfaceDescriptor(
        "rough", "custom", AerosolSurfaceRegime.VEGETATED_ROUGH, _prov("surface"), "grass A"
    )
    state = SurfaceState(descriptor, False, _prov("state"))
    gas = GasSurfaceParameterSet(
        GasPathParameterSet(500.0, 100.0, 300.0, _prov("gas")),
        StomatalParameterSet(_stomatal_params("grass B"), _prov("stomata")),
    )
    aerosol = RoughAerosolParameterSet(0.5, 1.0, 1e-3, 5e-6, _prov("aerosol"))
    with pytest.raises(ValueError):
        SurfacePhysicsBundle(state, gas, aerosol)


def test_rebound_provider_uses_explicit_surface_state_and_parameter_threshold():
    bundle = _rough_bundle(wet=False)
    provider = bundle.rebound_provider()
    assert provider.surface.is_wet is False
    assert provider.rebound_activation_diameter_m == 5.0e-6
    assert provider.sticking_fraction(stokes_number=4.0, particle_diameter_m=5.0e-6) == 1.0
    assert provider.sticking_fraction(stokes_number=4.0, particle_diameter_m=6.0e-6) < 1.0


def test_surface_abstraction_has_no_giltt_solver_or_land_use_numeric_mapping():
    import gilttpy.physics.surface_abstraction as sa

    source = inspect.getsource(sa)
    assert "ResolvedLowerInterface" not in source
    assert "steady_2d" not in source
    assert "transient_2d" not in source
    assert "land_use_parameters" not in source
    assert "LAND_USE" not in source
