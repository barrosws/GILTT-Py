import math
import pytest
from gilttpy.physics.deposition import (
    ConstantVelocity,GasResistance,AerosolResistanceSettling,
    cunningham_slip_correction,stokes_settling_velocity,ResolvedLowerInterface
)

def test_constant_velocity_exact():
    m=ConstantVelocity(0.02); assert m.deposition_velocity()==0.02 and m.downward_flux(3)==0.06

def test_gas_resistance_series():
    m=GasResistance(20,30,50); assert abs(m.deposition_velocity()-0.01)<1e-15

def test_gas_resistance_rejects_zero_total():
    with pytest.raises(ValueError): GasResistance(0,0,0)

def test_cunningham_is_above_one_and_larger_for_smaller_particle():
    assert cunningham_slip_correction(1e-8)>cunningham_slip_correction(1e-6)>1

def test_stokes_settling_positive():
    assert stokes_settling_velocity(1e-6,1500)>0

def test_stokes_near_quadratic_in_continuum_regime():
    v1=stokes_settling_velocity(20e-6,1500); v2=stokes_settling_velocity(40e-6,1500)
    assert 3.9 < v2/v1 < 4.1

def test_aerosol_zero_settling_limit():
    m=AerosolResistanceSettling(20,30,0); assert abs(m.deposition_velocity()-1/50)<1e-15

def test_aerosol_small_settling_continuity():
    a=AerosolResistanceSettling(20,30,1e-12).deposition_velocity(); b=1/50
    assert abs(a-b)/b < 1e-9

def test_flux_zero_at_zero_concentration():
    assert AerosolResistanceSettling(20,30,0.001).downward_flux(0)==0

def test_lower_interface_is_explicit_metadata():
    x=ResolvedLowerInterface(0.03,ConstantVelocity(0.01)); assert x.z_lower_m==0.03 and x.downward_flux(2)==0.02
