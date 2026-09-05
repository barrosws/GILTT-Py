import inspect
import math

import pytest

from gilttpy.physics.bidirectional_exchange import (
    BidirectionalExchangeResult,
    CanopyNodeExchange,
    CompensationPath,
    ExchangeRegime,
    NH3CompensationState,
    StandaloneNH3BidirectionalExchange,
    nh3_compensation_point_ug_m3,
)
from gilttpy.physics.deposition import GasResistance
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


def _prov(label):
    return ProvenanceRecord(label, "QA-only synthetic/reference audit", "QA033-v1")


def _path(label, r, chi):
    return CompensationPath(label, r, chi, _prov(label))


def _meteorology():
    return GasDepositionMeteorology(
        friction_velocity_m_s=0.4,
        temperature_k=298.15,
        pressure_pa=101325.0,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.01,
        monin_obukhov_length_m=math.inf,
        provenance=_prov("QA033 meteorology"),
    )


def _environment():
    return StomatalEnvironment(
        ppfd_umol_m2_s=800.0,
        temperature_c=25.0,
        vapor_pressure_deficit_kpa=1.0,
        soil_water_content_m3_m3=0.25,
        day_of_year=180.0,
        leaf_area_index=3.0,
    )


def _surface(*, stomatal=True, external_r=500.0, inc_r=100.0, soil_r=300.0):
    descriptor = SurfaceDescriptor(
        surface_label="QA033 reference surface",
        land_use_label="QA custom vegetation",
        aerosol_regime=AerosolSurfaceRegime.VEGETATED_ROUGH,
        provenance=_prov("QA033 surface identity"),
        vegetation_type="QA grass" if stomatal else None,
    )
    state = SurfaceState(descriptor, False, _prov("QA033 dry state"))
    paths = GasPathParameterSet(
        external_surface_resistance_s_m=external_r,
        in_canopy_resistance_s_m=inc_r,
        soil_resistance_s_m=soil_r,
        provenance=_prov("QA033 gas paths"),
    )
    stom = None
    if stomatal:
        pars = StomatalPhysiologyParameters(
            vegetation_type="QA grass",
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
        stom = StomatalParameterSet(pars, _prov("QA033 stomatal parameters"))
    gas = GasSurfaceParameterSet(paths=paths, stomatal=stom)
    aerosol = RoughAerosolParameterSet(0.5, 1.0, 2e-3, 5e-6, _prov("QA033 aerosol placeholder"))
    return SurfacePhysicsBundle(state, gas, aerosol)


def _nh3_species():
    return GasSpeciesDepositionProperties.from_massman_reference(
        "NH3",
        temperature_k=298.15,
        pressure_pa=101325.0,
        exchange_assumption=GasExchangeAssumption.BIDIRECTIONAL_REQUIRED,
        provenance=_prov("Massman NH3 reference"),
        stomatal_reference_species="NH3",
    )


def test_nh3_compensation_equilibrium_equation_exact_and_gamma_scaling():
    gamma = 1200.0; t = 25.0; tk = t + 273.15
    expected = (2.75e15 / tk) * math.exp(-1.04e4 / tk) * gamma
    assert nh3_compensation_point_ug_m3(emission_potential_gamma=gamma, temperature_c=t) == pytest.approx(expected, rel=1e-15)
    assert nh3_compensation_point_ug_m3(emission_potential_gamma=0.0, temperature_c=t) == 0.0
    assert nh3_compensation_point_ug_m3(emission_potential_gamma=2*gamma, temperature_c=t) == pytest.approx(2*expected, rel=1e-15)


def test_single_path_network_matches_exact_two_resistance_solution():
    node = CanopyNodeExchange(10.0, 20.0, 10.0, (_path("surface", 70.0, 2.0),))
    chi_c, flux_up, _, residual, regime = node.solve()
    expected_down = (10.0 - 2.0) / (20.0 + 10.0 + 70.0)
    assert flux_up == pytest.approx(-expected_down, rel=1e-15)
    assert chi_c == pytest.approx(10.0 - expected_down * 30.0, rel=1e-15)
    assert abs(residual) < 1e-14
    assert regime is ExchangeRegime.DEPOSITION


def test_exact_deposition_equilibrium_and_emission_regimes_under_upward_positive_sign():
    dep = CanopyNodeExchange(10.0, 10.0, 10.0, (_path("p", 20.0, 0.0),)).solve()
    eq = CanopyNodeExchange(10.0, 10.0, 10.0, (_path("p", 20.0, 10.0),)).solve()
    emi = CanopyNodeExchange(10.0, 10.0, 10.0, (_path("p", 20.0, 20.0),)).solve()
    assert dep[1] < 0 and dep[4] is ExchangeRegime.DEPOSITION
    assert eq[1] == pytest.approx(0.0, abs=1e-15) and eq[4] is ExchangeRegime.EQUILIBRIUM
    assert emi[1] > 0 and emi[4] is ExchangeRegime.EMISSION


def test_multichannel_node_is_mass_conservative_and_flux_equals_sum_of_pathway_sources():
    node = CanopyNodeExchange(
        8.0, 30.0, 20.0,
        (_path("stomatal", 100.0, 12.0), _path("external", 200.0, 0.0), _path("soil", 300.0, 20.0)),
    )
    chi_c, flux_up, path_fluxes, residual, _ = node.solve()
    assert 0.0 <= chi_c <= 20.0
    assert flux_up == pytest.approx(sum(p.flux_upward_ug_m2_s for p in path_fluxes), abs=1e-14)
    assert abs(residual) < 1e-14


def test_zero_compensation_multichannel_limit_recovers_qa032_gasresistance_exactly():
    ra, rb = 40.0, 15.0
    resistances = (100.0, 200.0, 300.0)
    node = CanopyNodeExchange(9.0, ra, rb, tuple(_path(str(i), r, 0.0) for i, r in enumerate(resistances)))
    _, flux_up, _, _, regime = node.solve()
    rc = 1.0 / sum(1.0/r for r in resistances)
    expected_down = GasResistance(ra, rb, rc).downward_flux(9.0)
    assert -flux_up == pytest.approx(expected_down, rel=1e-15)
    assert regime is ExchangeRegime.DEPOSITION


def test_closed_paths_are_ignored_and_all_closed_is_explicitly_isolated():
    a = CanopyNodeExchange(5.0, 20.0, 10.0, (_path("closed", math.inf, 100.0), _path("open", 50.0, 0.0))).solve()
    b = CanopyNodeExchange(5.0, 20.0, 10.0, (_path("open", 50.0, 0.0),)).solve()
    assert a[0] == pytest.approx(b[0], rel=1e-15)
    assert a[1] == pytest.approx(b[1], rel=1e-15)
    iso = CanopyNodeExchange(5.0, 20.0, 10.0, (_path("closed", math.inf, 100.0),)).solve()
    assert iso[0] == 5.0 and iso[1] == 0.0 and iso[4] is ExchangeRegime.ISOLATED


def test_high_level_nh3_provider_reuses_verified_ra_rb_and_surface_resistances():
    comp = NH3CompensationState(
        provenance=_prov("QA033 compensation state"),
        stomatal_gamma=1000.0, stomatal_temperature_c=25.0,
        external_gamma=0.0, external_temperature_c=25.0,
        soil_gamma=500.0, soil_temperature_c=20.0,
    )
    calc = StandaloneNH3BidirectionalExchange(
        _nh3_species(), _surface(), _meteorology(), comp, 8.0, _environment()
    )
    result = calc.result()
    assert isinstance(result, BidirectionalExchangeResult)
    assert result.species == "NH3"
    assert result.ra_s_m > 0.0 and result.rb_s_m > 0.0
    assert {p.label for p in result.pathway_fluxes} == {"stomatal", "external", "soil"}
    assert abs(result.mass_balance_residual_ug_m2_s) < 1e-13


def test_high_level_provider_requires_nh3_bidirectional_scope_and_explicit_active_path_potentials():
    wrong_scope = GasSpeciesDepositionProperties.from_massman_reference(
        "NH3", temperature_k=298.15, pressure_pa=101325.0,
        exchange_assumption=GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL,
        provenance=_prov("wrong scope"), stomatal_reference_species="NH3",
    )
    comp = NH3CompensationState(
        provenance=_prov("complete"),
        stomatal_gamma=0.0, stomatal_temperature_c=25.0,
        external_gamma=0.0, external_temperature_c=25.0,
        soil_gamma=0.0, soil_temperature_c=25.0,
    )
    with pytest.raises(ValueError, match="BIDIRECTIONAL_REQUIRED"):
        StandaloneNH3BidirectionalExchange(wrong_scope, _surface(), _meteorology(), comp, 8.0, _environment())
    incomplete = NH3CompensationState(
        provenance=_prov("incomplete"),
        stomatal_gamma=0.0, stomatal_temperature_c=25.0,
        external_gamma=0.0, external_temperature_c=25.0,
    )
    calc = StandaloneNH3BidirectionalExchange(_nh3_species(), _surface(), _meteorology(), incomplete, 8.0, _environment())
    with pytest.raises(ValueError, match="soil Gamma"):
        calc.result()


def test_high_level_zero_gamma_limit_matches_direct_unidirectional_chain_for_same_surface():
    surface = _surface(); met = _meteorology(); sp = _nh3_species(); env = _environment()
    comp = NH3CompensationState(
        provenance=_prov("zero gamma"),
        stomatal_gamma=0.0, stomatal_temperature_c=25.0,
        external_gamma=0.0, external_temperature_c=25.0,
        soil_gamma=0.0, soil_temperature_c=25.0,
    )
    result = StandaloneNH3BidirectionalExchange(sp, surface, met, comp, 10.0, env).result()
    rstom = surface.gas.stomatal_resistance_s_m(environment=env, species_to_reference_diffusivity_ratio=1.0)
    rw = surface.gas.paths.external_surface_resistance_s_m
    rsoil = surface.gas.paths.in_canopy_resistance_s_m + surface.gas.paths.soil_resistance_s_m
    rc = 1.0 / (1.0/rstom + 1.0/rw + 1.0/rsoil)
    expected_down = GasResistance(result.ra_s_m, result.rb_s_m, rc).downward_flux(10.0)
    assert result.downward_flux_ug_m2_s == pytest.approx(expected_down, rel=1e-14)


def test_qa033_has_no_giltt_boundary_import_and_keeps_empirical_gamma_mapping_outside_core():
    import gilttpy.physics.bidirectional_exchange as be
    source = inspect.getsource(be)
    assert "ResolvedLowerInterface" not in source
    assert "steady_2d" not in source
    assert "transient_2d" not in source
    assert "gamma_from_land_use" not in source.lower()
    assert "gamma_from_fertilization" not in source.lower()
    assert "long_term_nh3" not in source.lower()
