"""Source-tagged surface abstraction for GILTT-Py 2.0.

QA-031 connects the independently verified QA-030 physics providers without
introducing a land-use lookup table or a new deposition equation.  Surface
*identity* is metadata; numerical parameter sets remain explicit, immutable and
source-tagged.

The abstraction deliberately distinguishes:

- a free-form land-use/surface identity from aerosol collection regime;
- instantaneous wet/dry state from the identity of the surface;
- stomatal physiology parameters from non-stomatal gas-path resistances;
- vegetated/rough Zhang-2001 collection parameters from the smooth-surface branch.

No class label triggers hidden numerical defaults.  No GILTT solver or lower-
boundary object is imported here.  Transport-boundary coupling remains assigned
to QA-036/QA-038/QA-039.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Union

from .aerosol_collection import (
    ZHANG2001_MAX_BROWNIAN_EXPONENT,
    ZHANG2001_MIN_BROWNIAN_EXPONENT,
    Zhang2001SurfaceCollectionParameters,
)
from .aerosol_sticking import (
    AerosolSurfaceState,
    ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M,
    Zhang2001ReboundSticking,
)
from .stomatal import (
    JarvisEmbersonBulkStomatalResistance,
    StomatalEnvironment,
    StomatalPhysiologyParameters,
)
from .surface_resistance import DEPACCanopyResistance, ExplicitPathCanopyResistance


class AerosolSurfaceRegime(str, Enum):
    """Surface topology used by the verified Zhang-2001 collection providers."""

    VEGETATED_ROUGH = "vegetated_rough"
    SMOOTH = "smooth"


@dataclass(frozen=True)
class ProvenanceRecord:
    """Minimal source/applicability record for a surface or parameter set."""

    citation: str
    applicability: str
    version: str

    def __post_init__(self) -> None:
        if not str(self.citation).strip():
            raise ValueError("citation must be nonempty")
        if not str(self.applicability).strip():
            raise ValueError("applicability must be nonempty")
        if not str(self.version).strip():
            raise ValueError("version must be nonempty")

    @property
    def compact_label(self) -> str:
        return f"{self.citation} | {self.version} | {self.applicability}"


@dataclass(frozen=True)
class SurfaceDescriptor:
    """Identity of a surface, intentionally decoupled from numerical parameters."""

    surface_label: str
    land_use_label: str
    aerosol_regime: AerosolSurfaceRegime
    provenance: ProvenanceRecord
    vegetation_type: str | None = None

    def __post_init__(self) -> None:
        if not str(self.surface_label).strip():
            raise ValueError("surface_label must be nonempty")
        if not str(self.land_use_label).strip():
            raise ValueError("land_use_label must be nonempty")
        if not isinstance(self.aerosol_regime, AerosolSurfaceRegime):
            raise ValueError("aerosol_regime must be an AerosolSurfaceRegime")
        if self.vegetation_type is not None and not str(self.vegetation_type).strip():
            raise ValueError("vegetation_type must be nonempty when supplied")


@dataclass(frozen=True)
class SurfaceState:
    """Instantaneous surface state with provenance independent of land-use identity."""

    descriptor: SurfaceDescriptor
    is_wet: bool
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if type(self.is_wet) is not bool:
            raise ValueError("is_wet must be an explicit bool")

    def aerosol_surface_state(self) -> AerosolSurfaceState:
        return AerosolSurfaceState(
            surface_label=self.descriptor.surface_label,
            is_wet=self.is_wet,
            provenance=self.provenance.compact_label,
        )


@dataclass(frozen=True)
class StomatalParameterSet:
    """Source-tagged wrapper around the verified QA-030D physiology parameters."""

    parameters: StomatalPhysiologyParameters
    provenance: ProvenanceRecord



def _resistance(name: str, value: float) -> float:
    value = float(value)
    if math.isnan(value) or value < 0.0 or value == -math.inf:
        raise ValueError(f"{name} must be nonnegative or +inf")
    return value


@dataclass(frozen=True)
class GasPathParameterSet:
    """Explicit non-stomatal gas surface-path parameters.

    ``mesophyll_resistance_s_m=None`` selects the exact QA-030C DEPAC three-path
    topology.  Supplying a nonnegative value selects the explicitly separated
    stomatal+mesophyll series path.  ``None`` is therefore a topology selector,
    not a numerical resistance default.
    """

    external_surface_resistance_s_m: float
    in_canopy_resistance_s_m: float
    soil_resistance_s_m: float
    provenance: ProvenanceRecord
    mesophyll_resistance_s_m: float | None = None

    def __post_init__(self) -> None:
        _resistance("external_surface_resistance_s_m", self.external_surface_resistance_s_m)
        _resistance("in_canopy_resistance_s_m", self.in_canopy_resistance_s_m)
        _resistance("soil_resistance_s_m", self.soil_resistance_s_m)
        if self.mesophyll_resistance_s_m is not None:
            _resistance("mesophyll_resistance_s_m", self.mesophyll_resistance_s_m)

    def canopy_provider(self, *, stomatal_resistance_s_m: float):
        rstom = _resistance("stomatal_resistance_s_m", stomatal_resistance_s_m)
        if self.mesophyll_resistance_s_m is None:
            return DEPACCanopyResistance(
                stomatal_resistance_s_m=rstom,
                external_surface_resistance_s_m=self.external_surface_resistance_s_m,
                in_canopy_resistance_s_m=self.in_canopy_resistance_s_m,
                soil_resistance_s_m=self.soil_resistance_s_m,
            )
        return ExplicitPathCanopyResistance(
            stomatal_resistance_s_m=rstom,
            external_surface_resistance_s_m=self.external_surface_resistance_s_m,
            in_canopy_resistance_s_m=self.in_canopy_resistance_s_m,
            soil_resistance_s_m=self.soil_resistance_s_m,
            mesophyll_resistance_s_m=self.mesophyll_resistance_s_m,
        )


@dataclass(frozen=True)
class GasSurfaceParameterSet:
    """Surface-level gas parameter bundle with optional stomatal physiology."""

    paths: GasPathParameterSet
    stomatal: StomatalParameterSet | None = None

    def stomatal_resistance_s_m(
        self,
        *,
        environment: StomatalEnvironment | None,
        species_to_reference_diffusivity_ratio: float | None = None,
    ) -> float:
        if self.stomatal is None:
            if environment is not None:
                raise ValueError(
                    "environment supplied but no stomatal parameter set is configured"
                )
            return math.inf
        if environment is None:
            raise ValueError("stomatal environment is required for a vegetated stomatal path")
        if species_to_reference_diffusivity_ratio is None:
            raise ValueError("species_to_reference_diffusivity_ratio must be explicit for a stomatal path")
        return JarvisEmbersonBulkStomatalResistance(
            environment=environment,
            parameters=self.stomatal.parameters,
            species_to_reference_diffusivity_ratio=species_to_reference_diffusivity_ratio,
        ).resistance_s_m()

    def canopy_provider(
        self,
        *,
        environment: StomatalEnvironment | None,
        species_to_reference_diffusivity_ratio: float | None = None,
    ):
        rstom = self.stomatal_resistance_s_m(
            environment=environment,
            species_to_reference_diffusivity_ratio=species_to_reference_diffusivity_ratio,
        )
        return self.paths.canopy_provider(stomatal_resistance_s_m=rstom)


@dataclass(frozen=True)
class RoughAerosolParameterSet:
    """Source-tagged Zhang-2001 vegetated/rough collector parameters."""

    brownian_exponent: float
    impaction_alpha: float
    collector_radius_m: float
    rebound_activation_diameter_m: float
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        gamma = float(self.brownian_exponent)
        if not math.isfinite(gamma) or not (
            ZHANG2001_MIN_BROWNIAN_EXPONENT
            <= gamma
            <= ZHANG2001_MAX_BROWNIAN_EXPONENT
        ):
            raise ValueError("brownian_exponent must lie in [1/2, 2/3]")
        for name, value in (
            ("impaction_alpha", self.impaction_alpha),
            ("collector_radius_m", self.collector_radius_m),
            ("rebound_activation_diameter_m", self.rebound_activation_diameter_m),
        ):
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def collection_parameters(self, *, surface_label: str) -> Zhang2001SurfaceCollectionParameters:
        return Zhang2001SurfaceCollectionParameters(
            surface_label=surface_label,
            brownian_exponent=self.brownian_exponent,
            impaction_alpha=self.impaction_alpha,
            collector_radius_m=self.collector_radius_m,
            provenance=self.provenance.compact_label,
        )


@dataclass(frozen=True)
class SmoothAerosolParameterSet:
    """Source-tagged parameters needed by the current smooth Zhang/Slinn branch."""

    brownian_exponent: float
    rebound_activation_diameter_m: float
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        gamma = float(self.brownian_exponent)
        if not math.isfinite(gamma) or not (
            ZHANG2001_MIN_BROWNIAN_EXPONENT
            <= gamma
            <= ZHANG2001_MAX_BROWNIAN_EXPONENT
        ):
            raise ValueError("brownian_exponent must lie in [1/2, 2/3]")
        threshold = float(self.rebound_activation_diameter_m)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("rebound_activation_diameter_m must be finite and positive")


AerosolParameterSet = Union[RoughAerosolParameterSet, SmoothAerosolParameterSet]


@dataclass(frozen=True)
class SurfacePhysicsBundle:
    """Typed bridge from surface metadata to verified QA-030 provider inputs.

    This class performs compatibility checks only.  It contains no land-use to
    parameter mapping and imports no GILTT transport solver.
    """

    state: SurfaceState
    gas: GasSurfaceParameterSet
    aerosol: AerosolParameterSet

    def __post_init__(self) -> None:
        regime = self.state.descriptor.aerosol_regime
        if regime is AerosolSurfaceRegime.VEGETATED_ROUGH:
            if not isinstance(self.aerosol, RoughAerosolParameterSet):
                raise ValueError("VEGETATED_ROUGH surface requires RoughAerosolParameterSet")
        elif regime is AerosolSurfaceRegime.SMOOTH:
            if not isinstance(self.aerosol, SmoothAerosolParameterSet):
                raise ValueError("SMOOTH surface requires SmoothAerosolParameterSet")
        else:  # defensive against future enum extension
            raise ValueError(f"unsupported aerosol surface regime: {regime}")

        if self.gas.stomatal is not None:
            vegetation = self.state.descriptor.vegetation_type
            if vegetation is None:
                raise ValueError("stomatal parameters require descriptor.vegetation_type")
            if self.gas.stomatal.parameters.vegetation_type != vegetation:
                raise ValueError(
                    "descriptor vegetation_type must match stomatal parameter vegetation_type"
                )

    def aerosol_surface_state(self) -> AerosolSurfaceState:
        return self.state.aerosol_surface_state()

    def rebound_provider(self) -> Zhang2001ReboundSticking:
        return Zhang2001ReboundSticking(
            surface=self.aerosol_surface_state(),
            rebound_activation_diameter_m=self.aerosol.rebound_activation_diameter_m,
            provenance=self.aerosol.provenance.compact_label,
        )

    def rough_collection_parameters(self) -> Zhang2001SurfaceCollectionParameters:
        if not isinstance(self.aerosol, RoughAerosolParameterSet):
            raise ValueError("rough collection parameters are unavailable for a smooth surface")
        return self.aerosol.collection_parameters(
            surface_label=self.state.descriptor.surface_label
        )


# Kept as a named constant only to make the source-family default visible; QA-031
# parameter dataclasses still require callers to pass the activation threshold.
SOURCE_REFERENCE_REBOUND_THRESHOLD_M = ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M
