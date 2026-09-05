import dataclasses
import math
import pytest

from gilttpy.physics.stomatal import (
    JarvisEmbersonBulkStomatalResistance,
    StomatalEnvironment,
    StomatalPhysiologyParameters,
    cardinal_temperature_factor,
    depac_stomatal_resistance_from_factors,
    diffusivity_ratio,
    emberson_light_factor,
    fixed_day_phenology_factor,
    linear_soil_water_content_factor,
    linear_vpd_factor,
    ppfd_from_shortwave_w_m2,
    vapor_pressure_deficit_kpa,
)


def params(**overrides):
    base = dict(
        vegetation_type="QA_REFERENCE_VEGETATION",
        maximum_leaf_conductance_m_s=0.010,
        minimum_fraction=0.10,
        light_response_per_umol=0.005,
        temperature_minimum_c=0.0,
        temperature_optimum_c=20.0,
        temperature_maximum_c=40.0,
        vpd_full_open_kpa=0.5,
        vpd_minimum_open_kpa=3.0,
        soil_wilting_point_m3_m3=0.10,
        soil_field_capacity_m3_m3=0.30,
        phenology_start_day=100.0,
        phenology_end_day=300.0,
        phenology_ramp_up_days=20.0,
        phenology_ramp_down_days=40.0,
        phenology_start_fraction=0.2,
        phenology_end_fraction=0.4,
        phenology_outside_fraction=0.0,
    )
    base.update(overrides)
    return StomatalPhysiologyParameters(**base)


def env(**overrides):
    base = dict(
        ppfd_umol_m2_s=1000.0,
        temperature_c=20.0,
        vapor_pressure_deficit_kpa=0.5,
        soil_water_content_m3_m3=0.30,
        day_of_year=180.0,
        leaf_area_index=3.0,
    )
    base.update(overrides)
    return StomatalEnvironment(**base)


def test_emberson_light_factor_exact_zero_monotonic_and_reference_value():
    assert emberson_light_factor(0.0, light_response_per_umol=0.005) == 0.0
    low = emberson_light_factor(100.0, light_response_per_umol=0.005)
    high = emberson_light_factor(1000.0, light_response_per_umol=0.005)
    assert 0.0 < low < high < 1.0
    assert high == pytest.approx(1.0 - math.exp(-5.0), rel=2e-15)


def test_cardinal_temperature_factor_has_exact_cardinal_limits_and_optimum():
    kw = dict(minimum_c=0.0, optimum_c=20.0, maximum_c=40.0)
    assert cardinal_temperature_factor(0.0, **kw) == 0.0
    assert cardinal_temperature_factor(20.0, **kw) == pytest.approx(1.0, rel=0, abs=0)
    assert cardinal_temperature_factor(40.0, **kw) == 0.0
    assert 0.0 < cardinal_temperature_factor(10.0, **kw) < 1.0


def test_vpd_factor_uses_unambiguous_full_open_and_minimum_open_endpoints():
    kw = dict(full_open_kpa=0.5, minimum_open_kpa=3.0, minimum_fraction=0.1)
    assert linear_vpd_factor(0.2, **kw) == 1.0
    assert linear_vpd_factor(3.5, **kw) == 0.1
    assert linear_vpd_factor(1.75, **kw) == pytest.approx(0.55, rel=1e-15)


def test_soil_water_factor_hits_wilting_floor_and_field_capacity_one():
    kw = dict(wilting_point_m3_m3=0.10, field_capacity_m3_m3=0.30, minimum_fraction=0.1)
    assert linear_soil_water_content_factor(0.05, **kw) == 0.1
    assert linear_soil_water_content_factor(0.30, **kw) == 1.0
    assert linear_soil_water_content_factor(0.20, **kw) == pytest.approx(0.5, rel=2e-15)


def test_fixed_day_phenology_ramps_are_exact_and_zero_outside_season():
    kw = dict(
        start_day=100.0,
        end_day=300.0,
        ramp_up_days=20.0,
        ramp_down_days=40.0,
        start_fraction=0.2,
        end_fraction=0.4,
        outside_fraction=0.0,
    )
    assert fixed_day_phenology_factor(99.0, **kw) == 0.0
    assert fixed_day_phenology_factor(100.0, **kw) == pytest.approx(0.2)
    assert fixed_day_phenology_factor(110.0, **kw) == pytest.approx(0.6)
    assert fixed_day_phenology_factor(180.0, **kw) == 1.0
    assert fixed_day_phenology_factor(280.0, **kw) == pytest.approx(0.7)
    assert fixed_day_phenology_factor(301.0, **kw) == 0.0


def test_radiation_and_humidity_helpers_match_frozen_conventions():
    assert ppfd_from_shortwave_w_m2(400.0) == pytest.approx(900.0, rel=0, abs=0)
    vpd = vapor_pressure_deficit_kpa(temperature_c=20.0, relative_humidity_fraction=0.50)
    expected_es = 0.61 * math.exp(17.502 * 20.0 / (240.97 + 20.0))
    assert vpd == pytest.approx(0.5 * expected_es, rel=2e-15)


def test_depac_factor_product_and_diffusivity_scaling_are_exact():
    ratio = diffusivity_ratio(species_diffusivity_m2_s=1.5e-5, reference_diffusivity_m2_s=2.0e-5)
    r = depac_stomatal_resistance_from_factors(
        maximum_leaf_conductance_m_s=0.01,
        leaf_area_index=2.0,
        phenology_factor=0.8,
        soil_water_factor=0.5,
        vpd_factor=0.75,
        temperature_factor=1.0,
        par_factor=0.9,
        species_to_reference_diffusivity_ratio=ratio,
    )
    expected_g = 0.01 * 2.0 * 0.75 * 0.8 * 0.5 * 0.75 * 1.0 * 0.9
    assert ratio == pytest.approx(0.75, rel=0, abs=0)
    assert r == pytest.approx(1.0 / expected_g, rel=2e-15)


def test_jarvis_emberson_floor_applies_to_t_vpd_soil_but_darkness_still_closes():
    p = params()
    stressed = JarvisEmbersonBulkStomatalResistance(
        env(temperature_c=-5.0, vapor_pressure_deficit_kpa=5.0, soil_water_content_m3_m3=0.05), p
    )
    assert stressed.environmental_floor_product == pytest.approx(p.minimum_fraction)
    assert stressed.resistance_s_m() < math.inf

    dark = JarvisEmbersonBulkStomatalResistance(
        env(ppfd_umol_m2_s=0.0, temperature_c=-5.0, vapor_pressure_deficit_kpa=5.0, soil_water_content_m3_m3=0.05), p
    )
    assert dark.light_factor == 0.0
    assert dark.resistance_s_m() == math.inf


def test_lai_and_gas_diffusivity_scaling_have_exact_resistance_limits():
    p = params(phenology_start_day=1.0, phenology_end_day=366.0, phenology_ramp_up_days=0, phenology_ramp_down_days=0)
    e = env(ppfd_umol_m2_s=10000.0)
    a = JarvisEmbersonBulkStomatalResistance(e, p, species_to_reference_diffusivity_ratio=1.0)
    b = JarvisEmbersonBulkStomatalResistance(e, p, species_to_reference_diffusivity_ratio=0.5)
    assert b.resistance_s_m() / a.resistance_s_m() == pytest.approx(2.0, rel=2e-15)
    zero_lai = JarvisEmbersonBulkStomatalResistance(env(leaf_area_index=0.0), p)
    assert zero_lai.resistance_s_m() == math.inf


def test_parameter_contract_is_explicit_and_invalid_states_fail():
    names = {f.name for f in dataclasses.fields(StomatalPhysiologyParameters)}
    assert "vegetation_type" in names and "maximum_leaf_conductance_m_s" in names
    assert "temperature_optimum_c" in names and "soil_field_capacity_m3_m3" in names
    with pytest.raises(ValueError):
        params(vegetation_type="")
    with pytest.raises(ValueError):
        params(temperature_minimum_c=25.0, temperature_optimum_c=20.0)
    with pytest.raises(ValueError):
        linear_vpd_factor(1.0, full_open_kpa=2.0, minimum_open_kpa=1.0)
    with pytest.raises(ValueError):
        fixed_day_phenology_factor(180, start_day=300, end_day=100)
    with pytest.raises(ValueError):
        env(day_of_year=367)
    with pytest.raises(ValueError):
        JarvisEmbersonBulkStomatalResistance(env(), params(), species_to_reference_diffusivity_ratio=0.0)
