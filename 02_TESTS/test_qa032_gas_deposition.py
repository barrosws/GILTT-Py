import inspect
import math

import pytest

from gilttpy.physics.aerodynamic import most_aerodynamic_resistance
from gilttpy.physics.deposition import GasResistance
from gilttpy.physics.gas_deposition import (
    GasDepositionMeteorology,
    GasExchangeAssumption,
    GasSpeciesDepositionProperties,
    StandaloneUnidirectionalGasDeposition,
)
from gilttpy.physics.quasi_laminar import (
    air_kinematic_viscosity_sutherland_m2_s,
    depac_gas_quasi_laminar_resistance,
    massman_1998_air_diffusivity_m2_s,
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
    return ProvenanceRecord(label, "QA-only synthetic/reference audit", "QA032-v1")


def _meteorology():
    return GasDepositionMeteorology(
        friction_velocity_m_s=0.4,
        temperature_k=298.15,
        pressure_pa=101325.0,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.01,
        displacement_height_m=0.0,
        monin_obukhov_length_m=math.inf,
        provenance=_prov("QA032 meteorology"),
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


def _surface(*, land_use="QA custom vegetation", stomatal=True, closed=False):
    descriptor = SurfaceDescriptor(
        surface_label="QA032 reference surface",
        land_use_label=land_use,
        aerosol_regime=AerosolSurfaceRegime.VEGETATED_ROUGH,
        provenance=_prov("QA032 surface identity"),
        vegetation_type="QA grass" if stomatal else None,
    )
    state = SurfaceState(descriptor, False, _prov("QA032 dry state"))
    r = math.inf if closed else 500.0
    paths = GasPathParameterSet(
        external_surface_resistance_s_m=r,
        in_canopy_resistance_s_m=math.inf if closed else 100.0,
        soil_resistance_s_m=math.inf if closed else 300.0,
        provenance=_prov("QA032 gas path resistances"),
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
        stom = StomatalParameterSet(pars, _prov("QA032 stomatal parameters"))
    gas = GasSurfaceParameterSet(paths=paths, stomatal=stom)
    aerosol = RoughAerosolParameterSet(0.5, 1.0, 2e-3, 5e-6, _prov("QA032 aerosol placeholder"))
    return SurfacePhysicsBundle(state, gas, aerosol)


def _species(name="O3", *, reference="O3", assumption=GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL):
    return GasSpeciesDepositionProperties.from_massman_reference(
        name,
        temperature_k=298.15,
        pressure_pa=101325.0,
        exchange_assumption=assumption,
        provenance=_prov(f"Massman reference for {name}"),
        stomatal_reference_species=reference,
    )


def test_full_chain_matches_independently_composed_ra_rb_rc_and_gasresistance():
    met = _meteorology(); surface = _surface(); sp = _species("O3", reference="O3")
    calc = StandaloneUnidirectionalGasDeposition(sp, surface, met, _environment())
    result = calc.result()
    ra = most_aerodynamic_resistance(
        friction_velocity_m_s=0.4, reference_height_m=10.0,
        scalar_roughness_length_m=0.01, monin_obukhov_length_m=math.inf,
    )
    nu = air_kinematic_viscosity_sutherland_m2_s(temperature_k=298.15, pressure_pa=101325.0)
    rb = depac_gas_quasi_laminar_resistance(
        friction_velocity_m_s=0.4, molecular_diffusivity_m2_s=sp.molecular_diffusivity_m2_s,
        air_kinematic_viscosity_m2_s=nu,
    )
    rc = surface.gas.canopy_provider(
        environment=_environment(), species_to_reference_diffusivity_ratio=1.0
    ).resistance_s_m()
    assert result.ra_s_m == pytest.approx(ra, rel=1e-15)
    assert result.rb_s_m == pytest.approx(rb, rel=1e-15)
    assert result.rc_s_m == pytest.approx(rc, rel=1e-15)
    assert result.deposition_velocity_m_s == pytest.approx(GasResistance(ra, rb, rc).deposition_velocity(), rel=1e-15)


def test_zero_surface_concentration_flux_is_linear_and_zero_at_zero_concentration():
    calc = StandaloneUnidirectionalGasDeposition(_species(), _surface(), _meteorology(), _environment())
    assert calc.downward_flux(0.0) == 0.0
    assert calc.downward_flux(20.0) == pytest.approx(2.0 * calc.downward_flux(10.0), rel=1e-15)
    with pytest.raises(ValueError):
        calc.downward_flux(-1.0)


def test_nh3_is_hard_guarded_from_qa032_unidirectional_scope():
    sp = _species("NH3", reference="NH3")
    with pytest.raises(ValueError, match="QA-033"):
        StandaloneUnidirectionalGasDeposition(sp, _surface(), _meteorology(), _environment())


def test_non_unidirectional_exchange_declarations_are_rejected():
    for assumption in (GasExchangeAssumption.BIDIRECTIONAL_REQUIRED, GasExchangeAssumption.UNRESOLVED):
        sp = _species("O3", reference="O3", assumption=assumption)
        with pytest.raises(ValueError):
            StandaloneUnidirectionalGasDeposition(sp, _surface(), _meteorology(), _environment())


def test_massman_species_and_reference_diffusivities_produce_explicit_stomatal_ratio():
    sp = _species("SO2", reference="O3")
    expected = massman_1998_air_diffusivity_m2_s("SO2", temperature_k=298.15, pressure_pa=101325.0) / massman_1998_air_diffusivity_m2_s("O3", temperature_k=298.15, pressure_pa=101325.0)
    assert sp.stomatal_diffusivity_ratio == pytest.approx(expected, rel=1e-15)
    assert sp.stomatal_diffusivity_ratio != 1.0


def test_stomatal_surface_requires_reference_diffusivity_and_environment():
    d = massman_1998_air_diffusivity_m2_s("O3", temperature_k=298.15, pressure_pa=101325.0)
    sp = GasSpeciesDepositionProperties(
        "O3", d, GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL, _prov("explicit O3"), None
    )
    with pytest.raises(ValueError, match="stomatal_reference_diffusivity"):
        StandaloneUnidirectionalGasDeposition(sp, _surface(), _meteorology(), _environment())
    sp2 = _species()
    with pytest.raises(ValueError, match="stomatal_environment"):
        StandaloneUnidirectionalGasDeposition(sp2, _surface(), _meteorology(), None)


def test_nonstomatal_surface_needs_no_reference_diffusivity_and_closed_surface_gives_zero_vd():
    d = massman_1998_air_diffusivity_m2_s("NO2", temperature_k=298.15, pressure_pa=101325.0)
    sp = GasSpeciesDepositionProperties(
        "NO2", d, GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL, _prov("explicit NO2"), None
    )
    calc = StandaloneUnidirectionalGasDeposition(sp, _surface(stomatal=False, closed=True), _meteorology(), None)
    result = calc.result()
    assert result.rc_s_m == math.inf
    assert result.deposition_velocity_m_s == 0.0
    assert result.downward_flux(100.0) == 0.0


def test_higher_species_diffusivity_lowers_rb_when_other_conditions_are_fixed():
    met = _meteorology(); surface = _surface(stomatal=False)
    slow = GasSpeciesDepositionProperties("slow", 1e-5, GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL, _prov("slow"))
    fast = GasSpeciesDepositionProperties("fast", 2e-5, GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL, _prov("fast"))
    rslow = StandaloneUnidirectionalGasDeposition(slow, surface, met).result()
    rfast = StandaloneUnidirectionalGasDeposition(fast, surface, met).result()
    assert rfast.rb_s_m < rslow.rb_s_m
    assert rfast.deposition_velocity_m_s > rslow.deposition_velocity_m_s


def test_land_use_label_remains_numerically_inert_through_full_qa032_chain():
    sp = _species(); met = _meteorology(); env = _environment()
    a = StandaloneUnidirectionalGasDeposition(sp, _surface(land_use="class A"), met, env).result()
    b = StandaloneUnidirectionalGasDeposition(sp, _surface(land_use="unrelated class B"), met, env).result()
    assert a.ra_s_m == b.ra_s_m
    assert a.rb_s_m == b.rb_s_m
    assert a.rc_s_m == b.rc_s_m
    assert a.deposition_velocity_m_s == b.deposition_velocity_m_s


def test_qa032_has_no_giltt_boundary_or_bidirectional_compensation_implementation():
    import gilttpy.physics.gas_deposition as gd
    source = inspect.getsource(gd)
    assert "ResolvedLowerInterface" not in source.replace("``ResolvedLowerInterface``", "")
    assert "steady_2d" not in source
    assert "transient_2d" not in source
    assert "compensation_point" not in source
    assert "chi_comp" not in source
