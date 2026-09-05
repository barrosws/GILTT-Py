import dataclasses
import math
import pytest

from gilttpy.physics.deposition import GasResistance
from gilttpy.physics.surface_resistance import (
    DEPACCanopyResistance,
    ExplicitPathCanopyResistance,
    parallel_resistance_s_m,
    series_resistance_s_m,
)


def test_depac_three_parallel_path_equation_exact():
    rc = DEPACCanopyResistance(
        stomatal_resistance_s_m=100.0,
        external_surface_resistance_s_m=200.0,
        in_canopy_resistance_s_m=50.0,
        soil_resistance_s_m=250.0,
    )
    expected = 1.0 / (1.0 / 100.0 + 1.0 / 200.0 + 1.0 / 300.0)
    assert rc.effective_soil_resistance_s_m == pytest.approx(300.0, rel=0, abs=0)
    assert rc.resistance_s_m() == pytest.approx(expected, rel=2e-15)


def test_series_and_parallel_helpers_match_electrical_analog_exactly():
    assert series_resistance_s_m(20.0, 30.0, 50.0) == pytest.approx(100.0, rel=0, abs=0)
    assert parallel_resistance_s_m(100.0, 200.0) == pytest.approx(200.0 / 3.0, rel=2e-15)


def test_positive_infinity_closes_one_path_without_large_finite_cap():
    rc = DEPACCanopyResistance(
        stomatal_resistance_s_m=math.inf,
        external_surface_resistance_s_m=200.0,
        in_canopy_resistance_s_m=50.0,
        soil_resistance_s_m=250.0,
    )
    expected = 1.0 / (1.0 / 200.0 + 1.0 / 300.0)
    assert rc.resistance_s_m() == pytest.approx(expected, rel=2e-15)


def test_all_surface_paths_closed_gives_infinite_rc_and_zero_deposition_velocity():
    rc = DEPACCanopyResistance(
        stomatal_resistance_s_m=math.inf,
        external_surface_resistance_s_m=math.inf,
        in_canopy_resistance_s_m=0.0,
        soil_resistance_s_m=math.inf,
    ).resistance_s_m()
    assert rc == math.inf
    gas = GasResistance(ra_s_m=20.0, rb_s_m=30.0, rc_s_m=rc)
    assert gas.deposition_velocity() == 0.0
    assert gas.downward_flux(5.0) == 0.0


def test_zero_resistance_surface_path_dominates_parallel_network_exactly():
    rc = DEPACCanopyResistance(
        stomatal_resistance_s_m=0.0,
        external_surface_resistance_s_m=math.inf,
        in_canopy_resistance_s_m=0.0,
        soil_resistance_s_m=math.inf,
    )
    assert rc.resistance_s_m() == 0.0
    assert rc.conductance_m_s == math.inf
    assert GasResistance(20.0, 30.0, 0.0).deposition_velocity() == pytest.approx(1.0 / 50.0)


def test_adding_an_open_parallel_uptake_path_can_only_lower_rc():
    base = DEPACCanopyResistance(100.0, math.inf, 0.0, math.inf).resistance_s_m()
    with_external = DEPACCanopyResistance(100.0, 500.0, 0.0, math.inf).resistance_s_m()
    with_external_and_soil = DEPACCanopyResistance(100.0, 500.0, 50.0, 450.0).resistance_s_m()
    assert with_external_and_soil < with_external < base


def test_mesophyll_is_explicit_series_resistance_not_hidden_in_stomatal_physiology():
    no_mes = ExplicitPathCanopyResistance(100.0, math.inf, 0.0, math.inf, 0.0)
    with_mes = ExplicitPathCanopyResistance(100.0, math.inf, 0.0, math.inf, 75.0)
    assert no_mes.stomatal_path_resistance_s_m == 100.0
    assert with_mes.stomatal_path_resistance_s_m == 175.0
    assert with_mes.resistance_s_m() == 175.0


def test_explicit_path_with_zero_mesophyll_is_exactly_depac_equivalent():
    depac = DEPACCanopyResistance(120.0, 400.0, 25.0, 275.0)
    explicit = ExplicitPathCanopyResistance(120.0, 400.0, 25.0, 275.0, 0.0)
    assert explicit.resistance_s_m() == pytest.approx(depac.resistance_s_m(), rel=0, abs=0)


def test_qa030c_provider_signature_contains_no_stomatal_physiology_inputs():
    names = {f.name for f in dataclasses.fields(DEPACCanopyResistance)}
    assert names == {
        "stomatal_resistance_s_m",
        "external_surface_resistance_s_m",
        "in_canopy_resistance_s_m",
        "soil_resistance_s_m",
    }
    forbidden = {"radiation", "par", "vpd", "temperature", "soil_moisture", "lai", "phenology"}
    assert not (names & forbidden)


def test_invalid_resistances_fail_but_positive_infinity_is_valid():
    for bad in (-1.0, -math.inf, math.nan):
        with pytest.raises(ValueError):
            parallel_resistance_s_m(100.0, bad)
    with pytest.raises(ValueError):
        series_resistance_s_m()
    with pytest.raises(ValueError):
        parallel_resistance_s_m()
    assert series_resistance_s_m(10.0, math.inf) == math.inf
