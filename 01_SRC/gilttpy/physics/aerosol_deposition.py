"""Standalone complete aerosol dry-deposition closures for GILTT-Py 2.0.

QA-034 composes the independently verified QA-030 aerosol components into
*complete, source-labelled model families* while preserving their algebraic
separation.  It does not construct a GILTT lower-boundary condition.

Two complete closure families are exposed:

1. Zhang et al. (2001) / Slinn resistance family::

       Rs = 1 / [epsilon0 * u* * (E_B + E_imp + E_int) * R1]
       Vd = Vg + 1 / (Ra + Rs)

   with ``epsilon0=3`` as the source-family value.  The vegetated/rough branch
   uses Brownian diffusion + Peters-Eiden impaction + interception.  The smooth
   branch uses Brownian diffusion + the Slinn smooth-surface impaction relation
   and does not silently import the collector-interception term.

2. Venkatram-Pleim (1999) mass-consistent settling/resistance family::

       Vd = Vg / {1 - exp[-Vg (Ra + Rb)]}

   evaluated through the already verified QA-029 stable ``expm1`` kernel.  The
   non-settling resistance ``Rb`` is a separate, provenance-carrying input; this
   module deliberately does not construct it by borrowing Zhang-2001 ``Rs``.

Both closures include gravitational settling explicitly in the *deposition
closure*.  Therefore neither may be inserted unchanged at a GILTT boundary if
QA-038 also resolves the same settling velocity inside the transport PDE.
That flux partition remains a QA-036/QA-038/QA-039 gate.

Primary source families:
- Zhang, Gong, Padro & Barrie (2001), Atmospheric Environment 35, 549-560,
  DOI 10.1016/S1352-2310(00)00326-5.
- Venkatram & Pleim (1999), mass-consistent particle deposition formulation,
  retained in later operational/review formulations.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .aerodynamic import MOSTAerodynamicResistance
from .aerosol_collection import (
    AerosolCollectionState,
    CollectionEfficiencies,
    Zhang2001CollectionEfficiencies,
    zhang2001_brownian_efficiency,
    zhang2001_smooth_impaction_efficiency,
)
from .aerosol_sticking import sticking_adjusted_collection_efficiency
from .deposition import AerosolResistanceSettling
from .particle_physics import (
    AerosolAirState,
    AerosolParticleProperties,
    AerosolParticleTransportState,
)
from .surface_abstraction import (
    AerosolSurfaceRegime,
    ProvenanceRecord,
    RoughAerosolParameterSet,
    SmoothAerosolParameterSet,
    SurfacePhysicsBundle,
)


ZHANG2001_EPSILON0 = 3.0
SETTLING_COUPLING_STATUS = (
    "STANDALONE_CLOSURE_INCLUDES_VG__HOLD_PDE_VS_BOUNDARY_PARTITION_QA036_QA038_QA039"
)


class AerosolDepositionModelFamily(str, Enum):
    """Explicit complete aerosol-deposition model family."""

    ZHANG2001_SLINN = "zhang2001_slinn"
    VENKATRAM_PLEIM_1999 = "venkatram_pleim_1999"


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def zhang2001_surface_resistance_s_m(
    *,
    friction_velocity_m_s: float,
    total_collection_efficiency: float,
    sticking_fraction: float,
    epsilon0: float = ZHANG2001_EPSILON0,
) -> float:
    """Return Zhang-2001 surface resistance ``Rs`` in s m-1.

    ``Rs = 1/[epsilon0*u*E_total*R1]``.  A zero collection/sticking product is
    represented by the exact closed-path limit ``+inf``; no large finite cap is
    introduced.
    """
    ustar = _positive("friction_velocity_m_s", friction_velocity_m_s)
    e_total = _nonnegative("total_collection_efficiency", total_collection_efficiency)
    eps0 = _positive("epsilon0", epsilon0)
    r1 = float(sticking_fraction)
    if not math.isfinite(r1) or not 0.0 <= r1 <= 1.0:
        raise ValueError("sticking_fraction must be finite and lie in [0, 1]")
    effective = sticking_adjusted_collection_efficiency(
        total_collection_efficiency=e_total,
        sticking_fraction=r1,
    )
    if effective == 0.0:
        return math.inf
    return 1.0 / (eps0 * ustar * effective)


def zhang2001_complete_deposition_velocity_m_s(
    *, settling_velocity_m_s: float, aerodynamic_resistance_s_m: float, surface_resistance_s_m: float
) -> float:
    """Return the complete Zhang-2001 dry-deposition velocity.

    ``Vd = Vg + 1/(Ra+Rs)``.  ``Rs=+inf`` is the exact no-collection limit,
    leaving gravitational settling only.
    """
    vg = _nonnegative("settling_velocity_m_s", settling_velocity_m_s)
    ra = _nonnegative("aerodynamic_resistance_s_m", aerodynamic_resistance_s_m)
    rs = float(surface_resistance_s_m)
    if math.isnan(rs) or rs < 0.0 or rs == -math.inf:
        raise ValueError("surface_resistance_s_m must be nonnegative or +inf")
    if math.isinf(rs):
        return vg
    if ra + rs <= 0.0:
        raise ValueError("Ra + Rs must be positive")
    return vg + 1.0 / (ra + rs)


@dataclass(frozen=True)
class AerosolDepositionMeteorology:
    """Meteorological/air state for standalone aerosol deposition."""

    friction_velocity_m_s: float
    temperature_k: float
    pressure_pa: float
    reference_height_m: float
    scalar_roughness_length_m: float
    provenance: ProvenanceRecord
    displacement_height_m: float = 0.0
    monin_obukhov_length_m: float = math.inf

    def __post_init__(self) -> None:
        _positive("friction_velocity_m_s", self.friction_velocity_m_s)
        _positive("temperature_k", self.temperature_k)
        _positive("pressure_pa", self.pressure_pa)
        _positive("reference_height_m", self.reference_height_m)
        _positive("scalar_roughness_length_m", self.scalar_roughness_length_m)
        d = float(self.displacement_height_m)
        if not math.isfinite(d) or d < 0.0:
            raise ValueError("displacement_height_m must be finite and nonnegative")
        L = float(self.monin_obukhov_length_m)
        if math.isnan(L) or L == 0.0:
            raise ValueError("monin_obukhov_length_m must be nonzero or +/-inf")
        self.aerodynamic_provider().resistance_s_m()

    def aerodynamic_provider(self) -> MOSTAerodynamicResistance:
        return MOSTAerodynamicResistance(
            friction_velocity_m_s=self.friction_velocity_m_s,
            reference_height_m=self.reference_height_m,
            scalar_roughness_length_m=self.scalar_roughness_length_m,
            displacement_height_m=self.displacement_height_m,
            monin_obukhov_length_m=self.monin_obukhov_length_m,
        )

    @property
    def air_state(self) -> AerosolAirState:
        return AerosolAirState(
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_pa,
        )


@dataclass(frozen=True)
class Zhang2001AerosolDepositionResult:
    """Auditable mechanism decomposition for one Zhang-2001 calculation."""

    model_family: AerosolDepositionModelFamily
    surface_regime: AerosolSurfaceRegime
    particle_diameter_m: float
    diameter_basis: str
    ra_s_m: float
    rs_s_m: float
    settling_velocity_m_s: float
    deposition_velocity_m_s: float
    schmidt_number: float
    stokes_number: float
    brownian_efficiency: float
    impaction_efficiency: float
    interception_efficiency: float
    sticking_fraction: float
    epsilon0: float
    particle_provenance: str
    surface_provenance: str
    meteorology_provenance: str
    settling_coupling_status: str = SETTLING_COUPLING_STATUS

    @property
    def total_collection_efficiency(self) -> float:
        return math.fsum(
            (self.brownian_efficiency, self.impaction_efficiency, self.interception_efficiency)
        )

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity_m_s * _nonnegative("concentration", concentration)


@dataclass(frozen=True)
class StandaloneZhang2001AerosolDeposition:
    """Complete standalone Zhang-2001/Slinn aerosol deposition closure."""

    particle: AerosolParticleProperties
    surface: SurfacePhysicsBundle
    meteorology: AerosolDepositionMeteorology
    epsilon0: float = ZHANG2001_EPSILON0

    def __post_init__(self) -> None:
        _positive("epsilon0", self.epsilon0)

    def _transport(self) -> AerosolParticleTransportState:
        return AerosolParticleTransportState(
            particle=self.particle,
            air=self.meteorology.air_state,
        )

    def _collection(self, transport: AerosolParticleTransportState) -> tuple[float, CollectionEfficiencies]:
        regime = self.surface.state.descriptor.aerosol_regime
        if regime is AerosolSurfaceRegime.VEGETATED_ROUGH:
            if not isinstance(self.surface.aerosol, RoughAerosolParameterSet):
                raise ValueError("rough aerosol regime requires RoughAerosolParameterSet")
            p = self.surface.rough_collection_parameters()
            st = transport.stokes_number(
                friction_velocity_m_s=self.meteorology.friction_velocity_m_s,
                surface_regime="vegetated",
                collector_radius_m=p.collector_radius_m,
            )
            eff = Zhang2001CollectionEfficiencies(
                state=AerosolCollectionState(
                    schmidt_number=transport.schmidt_number,
                    stokes_number=st,
                    particle_diameter_m=self.particle.diameter_m,
                ),
                surface=p,
            ).efficiencies()
            return st, eff
        if regime is AerosolSurfaceRegime.SMOOTH:
            if not isinstance(self.surface.aerosol, SmoothAerosolParameterSet):
                raise ValueError("smooth aerosol regime requires SmoothAerosolParameterSet")
            st = transport.stokes_number(
                friction_velocity_m_s=self.meteorology.friction_velocity_m_s,
                surface_regime="smooth",
            )
            eb = zhang2001_brownian_efficiency(
                schmidt_number=transport.schmidt_number,
                exponent=self.surface.aerosol.brownian_exponent,
            )
            eimp = zhang2001_smooth_impaction_efficiency(stokes_number=st)
            # No collector-radius interception term is imported into the smooth branch.
            return st, CollectionEfficiencies(eb, eimp, 0.0)
        raise ValueError(f"unsupported aerosol surface regime: {regime}")

    def result(self) -> Zhang2001AerosolDepositionResult:
        transport = self._transport()
        st, efficiencies = self._collection(transport)
        r1 = self.surface.rebound_provider().sticking_fraction(
            stokes_number=st,
            particle_diameter_m=self.particle.diameter_m,
        )
        rs = zhang2001_surface_resistance_s_m(
            friction_velocity_m_s=self.meteorology.friction_velocity_m_s,
            total_collection_efficiency=efficiencies.total,
            sticking_fraction=r1,
            epsilon0=self.epsilon0,
        )
        ra = self.meteorology.aerodynamic_provider().resistance_s_m()
        vg = transport.settling_velocity_m_s
        vd = zhang2001_complete_deposition_velocity_m_s(
            settling_velocity_m_s=vg,
            aerodynamic_resistance_s_m=ra,
            surface_resistance_s_m=rs,
        )
        return Zhang2001AerosolDepositionResult(
            model_family=AerosolDepositionModelFamily.ZHANG2001_SLINN,
            surface_regime=self.surface.state.descriptor.aerosol_regime,
            particle_diameter_m=self.particle.diameter_m,
            diameter_basis=self.particle.diameter_basis,
            ra_s_m=ra,
            rs_s_m=rs,
            settling_velocity_m_s=vg,
            deposition_velocity_m_s=vd,
            schmidt_number=transport.schmidt_number,
            stokes_number=st,
            brownian_efficiency=efficiencies.brownian,
            impaction_efficiency=efficiencies.impaction,
            interception_efficiency=efficiencies.interception,
            sticking_fraction=r1,
            epsilon0=float(self.epsilon0),
            particle_provenance=self.particle.provenance,
            surface_provenance=self.surface.aerosol.provenance.compact_label,
            meteorology_provenance=self.meteorology.provenance.compact_label,
        )

    def deposition_velocity_m_s(self) -> float:
        return self.result().deposition_velocity_m_s

    def downward_flux(self, concentration: float) -> float:
        return self.result().downward_flux(concentration)


@dataclass(frozen=True)
class VenkatramPleimNonsettlingResistance:
    """Source-tagged non-settling resistance for the VP-1999 closure.

    The resistance is intentionally supplied rather than synthesized from the
    Zhang-2001 surface-resistance provider.  This prevents silent construction
    of a hybrid cross-scheme model.  A future source-specific VP/Slinn/Pleim
    resistance provider can satisfy this contract after an independent audit.
    """

    resistance_s_m: float
    surface_label: str
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        _nonnegative("resistance_s_m", self.resistance_s_m)
        if not str(self.surface_label).strip():
            raise ValueError("surface_label must be nonempty")


@dataclass(frozen=True)
class VenkatramPleimAerosolDepositionResult:
    """Auditable complete Venkatram-Pleim settling/resistance result."""

    model_family: AerosolDepositionModelFamily
    particle_diameter_m: float
    diameter_basis: str
    ra_s_m: float
    rb_s_m: float
    settling_velocity_m_s: float
    deposition_velocity_m_s: float
    particle_provenance: str
    resistance_provenance: str
    meteorology_provenance: str
    settling_coupling_status: str = SETTLING_COUPLING_STATUS

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity_m_s * _nonnegative("concentration", concentration)


@dataclass(frozen=True)
class StandaloneVenkatramPleimAerosolDeposition:
    """Complete standalone Venkatram-Pleim (1999) settling/resistance closure."""

    particle: AerosolParticleProperties
    resistance: VenkatramPleimNonsettlingResistance
    meteorology: AerosolDepositionMeteorology

    def result(self) -> VenkatramPleimAerosolDepositionResult:
        transport = AerosolParticleTransportState(
            particle=self.particle,
            air=self.meteorology.air_state,
        )
        ra = self.meteorology.aerodynamic_provider().resistance_s_m()
        rb = float(self.resistance.resistance_s_m)
        vg = transport.settling_velocity_m_s
        vd = AerosolResistanceSettling(
            ra_s_m=ra,
            rb_s_m=rb,
            settling_velocity_m_s=vg,
        ).deposition_velocity()
        return VenkatramPleimAerosolDepositionResult(
            model_family=AerosolDepositionModelFamily.VENKATRAM_PLEIM_1999,
            particle_diameter_m=self.particle.diameter_m,
            diameter_basis=self.particle.diameter_basis,
            ra_s_m=ra,
            rb_s_m=rb,
            settling_velocity_m_s=vg,
            deposition_velocity_m_s=vd,
            particle_provenance=self.particle.provenance,
            resistance_provenance=self.resistance.provenance.compact_label,
            meteorology_provenance=self.meteorology.provenance.compact_label,
        )

    def deposition_velocity_m_s(self) -> float:
        return self.result().deposition_velocity_m_s

    def downward_flux(self, concentration: float) -> float:
        return self.result().downward_flux(concentration)
