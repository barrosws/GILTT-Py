"""Standalone unidirectional gas dry deposition for GILTT-Py 2.0.

QA-032 composes the already verified QA-030A/030B/030C/030D/031 providers
into a reference-height resistance calculation

    Vd = 1 / (Ra + Rb + Rc).

This module is intentionally *not* a GILTT lower-boundary closure.  In
particular it does not import ``ResolvedLowerInterface`` or any transport
solver.  The resolved-versus-parameterized aerodynamic-transfer partition
remains a QA-036 gate.

The zero-surface-concentration / pure-uptake assumption is explicit.  Species
that require compensation-point or other bidirectional exchange must not be
passed through this provider.  NH3 is guarded explicitly because contemporary
DEPAC documentation identifies compensation-point exchange as essential to
represent possible re-emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .aerodynamic import MOSTAerodynamicResistance
from .deposition import GasResistance
from .quasi_laminar import (
    DEFAULT_AIR_PRANDTL,
    MASSMAN_1998_AIR_DIFFUSIVITY,
    air_kinematic_viscosity_sutherland_m2_s,
    massman_1998_air_diffusivity_m2_s,
    DEPACGasQuasiLaminarResistance,
)
from .stomatal import StomatalEnvironment
from .surface_abstraction import ProvenanceRecord, SurfacePhysicsBundle


class GasExchangeAssumption(str, Enum):
    """Exchange scope declared for a gas deposition calculation."""

    ZERO_COMPENSATION_UNIDIRECTIONAL = "zero_compensation_unidirectional"
    BIDIRECTIONAL_REQUIRED = "bidirectional_required"
    UNRESOLVED = "unresolved"


# Explicit safety/science guard for the QA-032 scope.  This is deliberately not
# a universal chemical taxonomy; it records the species for which the present
# project has source-level evidence that a zero compensation point can be
# materially misleading and whose implementation is therefore assigned QA-033.
QA032_BIDIRECTIONAL_GUARD_SPECIES = frozenset({"NH3"})


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_concentration(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("concentration must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class GasSpeciesDepositionProperties:
    """Species transport data and exchange-scope provenance.

    ``molecular_diffusivity_m2_s`` is the diffusivity at the meteorological
    state used by the calculation.  If a stomatal pathway is active,
    ``stomatal_reference_diffusivity_m2_s`` must also be supplied so the
    species/reference conductance ratio is computed rather than guessed.
    """

    species: str
    molecular_diffusivity_m2_s: float
    exchange_assumption: GasExchangeAssumption
    provenance: ProvenanceRecord
    stomatal_reference_diffusivity_m2_s: float | None = None

    def __post_init__(self) -> None:
        if not str(self.species).strip():
            raise ValueError("species must be nonempty")
        _positive("molecular_diffusivity_m2_s", self.molecular_diffusivity_m2_s)
        if not isinstance(self.exchange_assumption, GasExchangeAssumption):
            raise ValueError("exchange_assumption must be a GasExchangeAssumption")
        if self.stomatal_reference_diffusivity_m2_s is not None:
            _positive(
                "stomatal_reference_diffusivity_m2_s",
                self.stomatal_reference_diffusivity_m2_s,
            )

    @property
    def normalized_species(self) -> str:
        return str(self.species).strip().upper()

    @property
    def stomatal_diffusivity_ratio(self) -> float | None:
        if self.stomatal_reference_diffusivity_m2_s is None:
            return None
        return self.molecular_diffusivity_m2_s / self.stomatal_reference_diffusivity_m2_s

    @classmethod
    def from_massman_reference(
        cls,
        species: str,
        *,
        temperature_k: float,
        pressure_pa: float,
        exchange_assumption: GasExchangeAssumption,
        provenance: ProvenanceRecord,
        stomatal_reference_species: str | None = None,
    ) -> "GasSpeciesDepositionProperties":
        """Construct a QA/reference property state from the Massman-1998 library.

        This helper does not infer exchange physics.  The caller must still
        declare ``exchange_assumption``.  A stomatal reference species is also
        explicit when the surface parameterization is referenced to another gas.
        """
        d_species = massman_1998_air_diffusivity_m2_s(
            species, temperature_k=temperature_k, pressure_pa=pressure_pa
        )
        d_ref = None
        if stomatal_reference_species is not None:
            d_ref = massman_1998_air_diffusivity_m2_s(
                stomatal_reference_species,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
            )
        return cls(
            species=species,
            molecular_diffusivity_m2_s=d_species,
            exchange_assumption=exchange_assumption,
            provenance=provenance,
            stomatal_reference_diffusivity_m2_s=d_ref,
        )


@dataclass(frozen=True)
class GasDepositionMeteorology:
    """Meteorological state for standalone reference-height gas deposition."""

    friction_velocity_m_s: float
    temperature_k: float
    pressure_pa: float
    reference_height_m: float
    scalar_roughness_length_m: float
    provenance: ProvenanceRecord
    displacement_height_m: float = 0.0
    monin_obukhov_length_m: float = math.inf
    prandtl_number: float = DEFAULT_AIR_PRANDTL

    def __post_init__(self) -> None:
        _positive("friction_velocity_m_s", self.friction_velocity_m_s)
        _positive("temperature_k", self.temperature_k)
        _positive("pressure_pa", self.pressure_pa)
        _positive("reference_height_m", self.reference_height_m)
        _positive("scalar_roughness_length_m", self.scalar_roughness_length_m)
        _positive("prandtl_number", self.prandtl_number)
        d = float(self.displacement_height_m)
        if not math.isfinite(d) or d < 0.0:
            raise ValueError("displacement_height_m must be finite and nonnegative")
        L = float(self.monin_obukhov_length_m)
        if math.isnan(L) or L == 0.0:
            raise ValueError("monin_obukhov_length_m must be nonzero or +/-inf")
        # Reuse the verified provider for all remaining geometry checks.
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
    def air_kinematic_viscosity_m2_s(self) -> float:
        return air_kinematic_viscosity_sutherland_m2_s(
            temperature_k=self.temperature_k, pressure_pa=self.pressure_pa
        )


@dataclass(frozen=True)
class GasDepositionResult:
    """Auditable decomposition of a standalone unidirectional calculation."""

    species: str
    ra_s_m: float
    rb_s_m: float
    rc_s_m: float
    deposition_velocity_m_s: float
    schmidt_number: float
    species_provenance: str
    surface_provenance: str
    meteorology_provenance: str

    @property
    def total_resistance_s_m(self) -> float:
        return self.ra_s_m + self.rb_s_m + self.rc_s_m

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity_m_s * _nonnegative_concentration(concentration)


@dataclass(frozen=True)
class StandaloneUnidirectionalGasDeposition:
    """Compose QA-030/031 providers into an auditable standalone gas Vd.

    The result is a reference-height resistance-model calculation only.  It is
    prohibited from being interpreted as the GILTT ``z_lower`` boundary law
    until QA-036 resolves the aerodynamic-transfer partition.
    """

    species: GasSpeciesDepositionProperties
    surface: SurfacePhysicsBundle
    meteorology: GasDepositionMeteorology
    stomatal_environment: StomatalEnvironment | None = None

    def __post_init__(self) -> None:
        if self.species.exchange_assumption is not GasExchangeAssumption.ZERO_COMPENSATION_UNIDIRECTIONAL:
            raise ValueError(
                "QA-032 accepts only ZERO_COMPENSATION_UNIDIRECTIONAL exchange; "
                "bidirectional/unresolved exchange belongs to QA-033 or later"
            )
        if self.species.normalized_species in QA032_BIDIRECTIONAL_GUARD_SPECIES:
            raise ValueError(
                f"{self.species.normalized_species} is guarded from QA-032 unidirectional use; "
                "use the QA-033 bidirectional exchange provider"
            )
        has_stomata = self.surface.gas.stomatal is not None
        if has_stomata and self.species.stomatal_diffusivity_ratio is None:
            raise ValueError(
                "stomatal_reference_diffusivity_m2_s is required when the surface has a stomatal pathway"
            )
        if (not has_stomata) and self.stomatal_environment is not None:
            raise ValueError("stomatal_environment supplied for a surface without a stomatal pathway")
        if has_stomata and self.stomatal_environment is None:
            raise ValueError("stomatal_environment is required for a stomatal surface")

    def _components(self) -> tuple[float, float, float, float]:
        ra = self.meteorology.aerodynamic_provider().resistance_s_m()
        rb_provider = DEPACGasQuasiLaminarResistance(
            friction_velocity_m_s=self.meteorology.friction_velocity_m_s,
            molecular_diffusivity_m2_s=self.species.molecular_diffusivity_m2_s,
            air_kinematic_viscosity_m2_s=self.meteorology.air_kinematic_viscosity_m2_s,
            prandtl_number=self.meteorology.prandtl_number,
        )
        rb = rb_provider.resistance_s_m()
        rc_provider = self.surface.gas.canopy_provider(
            environment=self.stomatal_environment,
            species_to_reference_diffusivity_ratio=self.species.stomatal_diffusivity_ratio,
        )
        rc = rc_provider.resistance_s_m()
        return ra, rb, rc, rb_provider.schmidt_number

    def result(self) -> GasDepositionResult:
        ra, rb, rc, sc = self._components()
        model = GasResistance(ra, rb, rc)
        return GasDepositionResult(
            species=self.species.normalized_species,
            ra_s_m=ra,
            rb_s_m=rb,
            rc_s_m=rc,
            deposition_velocity_m_s=model.deposition_velocity(),
            schmidt_number=sc,
            species_provenance=self.species.provenance.compact_label,
            surface_provenance=self.surface.state.descriptor.provenance.compact_label,
            meteorology_provenance=self.meteorology.provenance.compact_label,
        )

    def deposition_velocity_m_s(self) -> float:
        return self.result().deposition_velocity_m_s

    def downward_flux(self, concentration: float) -> float:
        return self.result().downward_flux(concentration)


# Exported only as a transparent reference to the current QA property coverage.
QA032_MASSMAN_REFERENCE_SPECIES = frozenset(MASSMAN_1998_AIR_DIFFUSIVITY)
