import inspect
import math

import pytest

from gilttpy.physics.aerosol_deposition import (
    AerosolDepositionMeteorology,
    AerosolDepositionModelFamily,
    StandaloneVenkatramPleimAerosolDeposition,
    StandaloneZhang2001AerosolDeposition,
    VenkatramPleimNonsettlingResistance,
)
from gilttpy.physics.aerosol_size_distribution import (
    AerosolDiameterDomain,
    AerosolDistributionWeighting,
    AerosolSizeDistribution,
    ConstantVenkatramPleimSizeResistance,
    LognormalAerosolMode,
    StandaloneAerosolSizeDistributionDeposition,
)
from gilttpy.physics.particle_physics import AerosolParticleProperties
from gilttpy.physics.surface_abstraction import (
    AerosolSurfaceRegime, GasPathParameterSet, GasSurfaceParameterSet,
    ProvenanceRecord, RoughAerosolParameterSet, SurfaceDescriptor,
    SurfacePhysicsBundle, SurfaceState,
)


def _prov(label): return ProvenanceRecord(label, "QA035 synthetic structural audit; not calibration", "QA-only")

def _domain(lo_um=0.01, hi_um=20.0): return AerosolDiameterDomain(lo_um*1e-6, hi_um*1e-6, _prov("diameter domain"))

def _surface():
    d=SurfaceDescriptor("QA rough","arbitrary",AerosolSurfaceRegime.VEGETATED_ROUGH,_prov("surface"))
    s=SurfaceState(d,False,_prov("dry state"))
    g=GasSurfaceParameterSet(GasPathParameterSet(500.0,100.0,300.0,_prov("gas stub")),None)
    a=RoughAerosolParameterSet(2/3,1.0,2e-3,5e-6,_prov("rough aerosol"))
    return SurfacePhysicsBundle(s,g,a)

def _met():
    return AerosolDepositionMeteorology(0.4,298.15,101325.0,10.0,0.01,_prov("meteorology"),monin_obukhov_length_m=math.inf)

def _mode(label="accum",dg_um=0.3,sg=1.8,n=1e8,rho=1500.0,basis="dry diameter"):
    return LognormalAerosolMode(label,n,dg_um*1e-6,sg,rho,basis,_prov(f"mode {label}"))

def _distribution(*m): return AerosolSizeDistribution(tuple(m),_prov("distribution"))


def test_lognormal_analytic_moments_and_mass_concentration_identity():
    m=_mode(dg_um=0.2,sg=1.7,n=2.5e8,rho=1800.0); s=math.log(1.7)
    for k in (0.,1.,2.,3.,5.):
        exp=(0.2e-6)**k*math.exp(0.5*k*k*s*s)
        assert m.normalized_moment_m_power(k)==pytest.approx(exp,rel=2e-15)
    assert m.total_mass_concentration_kg_m3==pytest.approx(2.5e8*1800*math.pi/6*m.normalized_moment_m_power(3),rel=2e-15)


def test_hatch_choate_number_to_mass_geometric_mean_shift():
    m=_mode(dg_um=0.1,sg=2.0)
    exp=0.1e-6*math.exp(3*math.log(2.0)**2)
    assert m.mass_geometric_mean_diameter_m==pytest.approx(exp,rel=1e-15)


def test_explicit_domain_reports_number_and_mass_tail_coverage_analytically():
    m=_mode(dg_um=1.0,sg=2.0)
    d=_domain(0.01,5.0)
    fn=m.retained_fraction(domain=d,weighting=AerosolDistributionWeighting.NUMBER)
    fm=m.retained_fraction(domain=d,weighting=AerosolDistributionWeighting.MASS)
    assert 0<fm<fn<1
    assert fn>0.98
    assert fm<0.75


def test_monodisperse_sigma_one_reduces_exactly_to_qa034_zhang_single_particle():
    m=_mode(dg_um=1.0,sg=1.0); r=StandaloneAerosolSizeDistributionDeposition(_distribution(m),_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met(),surface=_surface()).result()
    p=AerosolParticleProperties(1e-6,m.density_kg_m3,m.diameter_basis,"QA")
    single=StandaloneZhang2001AerosolDeposition(p,_surface(),_met()).deposition_velocity_m_s()
    assert r.number_weighted_vd_m_s==pytest.approx(single,rel=0,abs=0)
    assert r.mass_weighted_vd_m_s==pytest.approx(single,rel=0,abs=0)
    assert r.retained_number_fraction==1 and r.retained_mass_fraction==1


def test_monodisperse_sigma_one_reduces_exactly_to_qa034_vp_single_particle():
    m=_mode(dg_um=3.0,sg=1.0); provider=ConstantVenkatramPleimSizeResistance(120,"QA VP",_prov("VP resistance"))
    r=StandaloneAerosolSizeDistributionDeposition(_distribution(m),_domain(),AerosolDepositionModelFamily.VENKATRAM_PLEIM_1999,_met(),vp_resistance_provider=provider).result()
    p=AerosolParticleProperties(3e-6,m.density_kg_m3,m.diameter_basis,"QA")
    single=StandaloneVenkatramPleimAerosolDeposition(p,VenkatramPleimNonsettlingResistance(120,"QA VP",_prov("VP resistance")),_met()).deposition_velocity_m_s()
    assert r.number_weighted_vd_m_s==pytest.approx(single,rel=0,abs=0)
    assert r.mass_weighted_vd_m_s==pytest.approx(single,rel=0,abs=0)


def test_polydisperse_mass_number_and_geometric_mean_evaluations_are_not_interchangeable():
    m=_mode(dg_um=0.3,sg=1.8); r=StandaloneAerosolSizeDistributionDeposition(_distribution(m),_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met(),surface=_surface(),quadrature_order=64).result()
    p=AerosolParticleProperties(m.geometric_mean_diameter_m,m.density_kg_m3,m.diameter_basis,"QA")
    atdg=StandaloneZhang2001AerosolDeposition(p,_surface(),_met()).deposition_velocity_m_s()
    assert abs(r.number_weighted_vd_m_s-atdg)/atdg>0.02
    assert abs(r.mass_weighted_vd_m_s-r.number_weighted_vd_m_s)/r.number_weighted_vd_m_s>0.10


def test_split_log_quadrature_converges_across_dry_rebound_threshold():
    dist=_distribution(_mode(dg_um=3.0,sg=2.0))
    r64=StandaloneAerosolSizeDistributionDeposition(dist,_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met(),surface=_surface(),quadrature_order=64).result()
    r128=StandaloneAerosolSizeDistributionDeposition(dist,_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met(),surface=_surface(),quadrature_order=128).result()
    assert abs(r64.number_weighted_vd_m_s-r128.number_weighted_vd_m_s)/r128.number_weighted_vd_m_s<2e-8
    assert abs(r64.mass_weighted_vd_m_s-r128.mass_weighted_vd_m_s)/r128.mass_weighted_vd_m_s<2e-8


def test_multimodal_aggregation_uses_in_domain_number_and_mass_weights_exactly():
    m1=_mode("fine",0.15,1.5,9e8,1400); m2=_mode("coarse",4.0,1.5,2e6,2500)
    r=StandaloneAerosolSizeDistributionDeposition(_distribution(m1,m2),_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met(),surface=_surface()).result()
    en=sum(x.in_domain_number_concentration_m3*x.number_weighted_vd_m_s for x in r.mode_results)/r.in_domain_number_concentration_m3
    em=sum(x.in_domain_mass_concentration_kg_m3*x.mass_weighted_vd_m_s for x in r.mode_results)/r.in_domain_mass_concentration_kg_m3
    assert r.number_weighted_vd_m_s==pytest.approx(en,rel=1e-15)
    assert r.mass_weighted_vd_m_s==pytest.approx(em,rel=1e-15)


def test_mixed_dry_wet_bases_invalid_width_and_invalid_domain_are_rejected():
    with pytest.raises(ValueError): _mode(sg=0.99)
    with pytest.raises(ValueError): AerosolDiameterDomain(1e-6,1e-6,_prov("bad"))
    with pytest.raises(ValueError): _distribution(_mode("dry",basis="dry"),_mode("wet",basis="wet"))


def test_model_family_wiring_requires_explicit_domain_and_preserves_no_giltt_import():
    dist=_distribution(_mode())
    with pytest.raises(ValueError): StandaloneAerosolSizeDistributionDeposition(dist,_domain(),AerosolDepositionModelFamily.ZHANG2001_SLINN,_met())
    with pytest.raises(ValueError): StandaloneAerosolSizeDistributionDeposition(dist,_domain(),AerosolDepositionModelFamily.VENKATRAM_PLEIM_1999,_met(),surface=_surface())
    import gilttpy.physics.aerosol_size_distribution as mod
    src=inspect.getsource(mod)
    assert "ResolvedLowerInterface" not in src and "steady_2d" not in src and "transient_2d" not in src
