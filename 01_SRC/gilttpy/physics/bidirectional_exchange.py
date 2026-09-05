"""Standalone bidirectional NH3 surface-atmosphere exchange for GILTT-Py 2.0.

QA-033 extends the verified QA-032 standalone gas-deposition chain with explicit
compensation-point physics.  It remains a reference-height resistance-network
calculation and is intentionally *not* a GILTT lower-boundary closure.

The network is solved at a canopy node.  Let Rab = Ra + Rb connect the ambient
concentration chi_a to the canopy-node concentration chi_c, and let surface
pathway i connect chi_c to a compensation concentration chi_i through Ri.  The
steady node balance is

    (chi_a - chi_c)/Rab = sum_i (chi_c - chi_i)/Ri,

hence

    chi_c = [chi_a/Rab + sum_i chi_i/Ri]
            / [1/Rab + sum_i 1/Ri].

The signed exchange flux follows the contemporary NH3 convention used in the
comparison literature: upward/emission is positive and deposition is negative,

    F_up = (chi_c - chi_a)/Rab.

This formulation has an important exact reduction.  If every active surface
compensation point is zero, the surface pathways combine to Rc in parallel and

    F_down = -F_up = chi_a/(Ra + Rb + Rc),

which recovers the verified QA-032 unidirectional resistance model.

NH3 pathway compensation concentrations may be supplied explicitly or derived
from emission potential Gamma=[NH4+]/[H+] using the common equilibrium relation
summarized by Jongenelen et al. (2025):

    chi_i [ug m-3] = (2.75e15/T_K) * exp(-1.04e4/T_K) * Gamma_i.

QA-033 deliberately does not infer Gamma from land use, fertilization history,
long-term NH3 concentration, RH, pH or soil state.  Those empirical mappings are
model-family- and application-dependent and remain source-tagged future inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from .gas_deposition import (
    GasDepositionMeteorology,
    GasExchangeAssumption,
    GasSpeciesDepositionProperties,
)
from .quasi_laminar import DEPACGasQuasiLaminarResistance
from .stomatal import StomatalEnvironment
from .surface_abstraction import ProvenanceRecord, SurfacePhysicsBundle


NH3_EQUILIBRIUM_A = 2.75e15
NH3_EQUILIBRIUM_B_K = 1.04e4


class ExchangeRegime(str, Enum):
    """Net reference-height exchange regime under upward-positive convention."""

    DEPOSITION = "deposition"
    EQUILIBRIUM = "equilibrium"
    EMISSION = "emission"
    ISOLATED = "isolated"


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_resistance(name: str, value: float) -> float:
    value = float(value)
    if math.isnan(value) or value <= 0.0 or value == -math.inf:
        raise ValueError(f"{name} must be positive or +inf")
    return value


def nh3_compensation_point_ug_m3(*, emission_potential_gamma: float, temperature_c: float) -> float:
    """Return NH3 compensation concentration from Gamma and temperature.

    The equation is the Henry/dissociation/ideal-gas equilibrium relation used
    across the contemporary DEPAC, Massad and Zhang NH3 exchange families as
    summarized by Jongenelen et al. (2025).  ``Gamma`` is dimensionless and
    ``temperature_c`` is the pathway temperature in degC.
    """
    gamma = _finite_nonnegative("emission_potential_gamma", emission_potential_gamma)
    t_c = float(temperature_c)
    if not math.isfinite(t_c):
        raise ValueError("temperature_c must be finite")
    t_k = t_c + 273.15
    if t_k <= 0.0:
        raise ValueError("temperature_c must exceed absolute zero")
    if gamma == 0.0:
        return 0.0
    return (NH3_EQUILIBRIUM_A / t_k) * math.exp(-NH3_EQUILIBRIUM_B_K / t_k) * gamma


@dataclass(frozen=True)
class CompensationPath:
    """One source/sink pathway connected to the canopy node."""

    label: str
    resistance_s_m: float
    compensation_concentration_ug_m3: float
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("label must be nonempty")
        _positive_resistance("resistance_s_m", self.resistance_s_m)
        _finite_nonnegative(
            "compensation_concentration_ug_m3", self.compensation_concentration_ug_m3
        )

    @property
    def is_open(self) -> bool:
        return math.isfinite(float(self.resistance_s_m))

    @property
    def conductance_m_s(self) -> float:
        return 0.0 if not self.is_open else 1.0 / float(self.resistance_s_m)


@dataclass(frozen=True)
class NH3CompensationState:
    """Explicit NH3 emission potentials and pathway temperatures.

    ``None`` means that the corresponding physical pathway is not configured.
    A configured active resistance requires an explicit Gamma/temperature pair;
    Gamma=0 is the explicit pure-uptake limit and is never inferred silently.
    """

    provenance: ProvenanceRecord
    stomatal_gamma: float | None = None
    stomatal_temperature_c: float | None = None
    external_gamma: float | None = None
    external_temperature_c: float | None = None
    soil_gamma: float | None = None
    soil_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for prefix in ("stomatal", "external", "soil"):
            gamma = getattr(self, f"{prefix}_gamma")
            temp = getattr(self, f"{prefix}_temperature_c")
            if (gamma is None) != (temp is None):
                raise ValueError(f"{prefix} Gamma and temperature must be supplied together")
            if gamma is not None:
                _finite_nonnegative(f"{prefix}_gamma", gamma)
                if not math.isfinite(float(temp)) or float(temp) + 273.15 <= 0.0:
                    raise ValueError(f"{prefix}_temperature_c is invalid")

    def concentration(self, pathway: str) -> float | None:
        key = str(pathway).strip().lower()
        if key not in {"stomatal", "external", "soil"}:
            raise ValueError("pathway must be stomatal, external or soil")
        gamma = getattr(self, f"{key}_gamma")
        temp = getattr(self, f"{key}_temperature_c")
        if gamma is None:
            return None
        return nh3_compensation_point_ug_m3(
            emission_potential_gamma=float(gamma), temperature_c=float(temp)
        )


@dataclass(frozen=True)
class PathwayFlux:
    """Upward-positive flux from one surface reservoir into the canopy node."""

    label: str
    resistance_s_m: float
    compensation_concentration_ug_m3: float
    flux_upward_ug_m2_s: float
    provenance: str


@dataclass(frozen=True)
class BidirectionalExchangeResult:
    """Auditable reference-height bidirectional exchange result."""

    species: str
    atmospheric_concentration_ug_m3: float
    canopy_concentration_ug_m3: float
    ra_s_m: float
    rb_s_m: float
    flux_upward_ug_m2_s: float
    regime: ExchangeRegime
    pathway_fluxes: tuple[PathwayFlux, ...]
    mass_balance_residual_ug_m2_s: float
    species_provenance: str
    surface_provenance: str
    meteorology_provenance: str
    compensation_provenance: str

    @property
    def downward_flux_ug_m2_s(self) -> float:
        return -self.flux_upward_ug_m2_s

    @property
    def atmospheric_transfer_resistance_s_m(self) -> float:
        return self.ra_s_m + self.rb_s_m


@dataclass(frozen=True)
class CanopyNodeExchange:
    """Generic conservative canopy-node resistance network.

    This low-level object is species-agnostic.  It is useful for exact algebraic
    tests and for future source-family-specific compensation-point providers.
    """

    atmospheric_concentration_ug_m3: float
    ra_s_m: float
    rb_s_m: float
    pathways: tuple[CompensationPath, ...]

    def __post_init__(self) -> None:
        _finite_nonnegative(
            "atmospheric_concentration_ug_m3", self.atmospheric_concentration_ug_m3
        )
        _positive_resistance("ra_s_m", self.ra_s_m)
        _positive_resistance("rb_s_m", self.rb_s_m)
        if not isinstance(self.pathways, tuple):
            object.__setattr__(self, "pathways", tuple(self.pathways))
        labels = [p.label for p in self.pathways]
        if len(set(labels)) != len(labels):
            raise ValueError("pathway labels must be unique")

    @property
    def rab_s_m(self) -> float:
        return float(self.ra_s_m) + float(self.rb_s_m)

    def solve(self) -> tuple[float, float, tuple[PathwayFlux, ...], float, ExchangeRegime]:
        chi_a = float(self.atmospheric_concentration_ug_m3)
        rab = self.rab_s_m
        g_atm = 0.0 if math.isinf(rab) else 1.0 / rab
        open_paths = tuple(p for p in self.pathways if p.is_open)
        if not open_paths:
            return chi_a, 0.0, tuple(), 0.0, ExchangeRegime.ISOLATED
        if g_atm == 0.0:
            # No reference-height exchange is possible even if internal pathways exist.
            numerator = sum(p.conductance_m_s * p.compensation_concentration_ug_m3 for p in open_paths)
            denominator = sum(p.conductance_m_s for p in open_paths)
            chi_c = numerator / denominator
            path_fluxes = tuple(
                PathwayFlux(
                    p.label,
                    p.resistance_s_m,
                    p.compensation_concentration_ug_m3,
                    (p.compensation_concentration_ug_m3 - chi_c) / p.resistance_s_m,
                    p.provenance.compact_label,
                )
                for p in open_paths
            )
            residual = sum(p.flux_upward_ug_m2_s for p in path_fluxes)
            return chi_c, 0.0, path_fluxes, residual, ExchangeRegime.ISOLATED

        numerator = chi_a * g_atm + sum(
            p.conductance_m_s * p.compensation_concentration_ug_m3 for p in open_paths
        )
        denominator = g_atm + sum(p.conductance_m_s for p in open_paths)
        chi_c = numerator / denominator
        flux_up = (chi_c - chi_a) / rab
        path_fluxes = tuple(
            PathwayFlux(
                p.label,
                p.resistance_s_m,
                p.compensation_concentration_ug_m3,
                (p.compensation_concentration_ug_m3 - chi_c) / p.resistance_s_m,
                p.provenance.compact_label,
            )
            for p in open_paths
        )
        residual = flux_up - sum(p.flux_upward_ug_m2_s for p in path_fluxes)
        scale = max(1.0, abs(flux_up), *(abs(p.flux_upward_ug_m2_s) for p in path_fluxes))
        tol = 256.0 * math.ulp(scale)
        if flux_up > tol:
            regime = ExchangeRegime.EMISSION
        elif flux_up < -tol:
            regime = ExchangeRegime.DEPOSITION
        else:
            regime = ExchangeRegime.EQUILIBRIUM
        return chi_c, flux_up, path_fluxes, residual, regime


@dataclass(frozen=True)
class StandaloneNH3BidirectionalExchange:
    """Compose QA-030/031 transport/surface providers with explicit NH3 potentials."""

    species: GasSpeciesDepositionProperties
    surface: SurfacePhysicsBundle
    meteorology: GasDepositionMeteorology
    compensation: NH3CompensationState
    atmospheric_concentration_ug_m3: float
    stomatal_environment: StomatalEnvironment | None = None

    def __post_init__(self) -> None:
        if self.species.normalized_species != "NH3":
            raise ValueError("QA-033 high-level provider is currently source-scoped to NH3")
        if self.species.exchange_assumption is not GasExchangeAssumption.BIDIRECTIONAL_REQUIRED:
            raise ValueError("NH3 QA-033 requires BIDIRECTIONAL_REQUIRED exchange assumption")
        _finite_nonnegative(
            "atmospheric_concentration_ug_m3", self.atmospheric_concentration_ug_m3
        )
        if self.surface.gas.paths.mesophyll_resistance_s_m is not None:
            raise ValueError(
                "QA-033 NH3 topology does not mix the optional QA-030C mesophyll extension; "
                "use explicit stomatal/external/soil pathways"
            )
        has_stomata = self.surface.gas.stomatal is not None
        if has_stomata:
            if self.stomatal_environment is None:
                raise ValueError("stomatal_environment is required for an active stomatal pathway")
            if self.species.stomatal_diffusivity_ratio is None:
                raise ValueError(
                    "stomatal_reference_diffusivity_m2_s is required for active stomatal exchange"
                )
        elif self.stomatal_environment is not None:
            raise ValueError("stomatal_environment supplied for a surface without stomata")

    def _ra_rb(self) -> tuple[float, float]:
        ra = self.meteorology.aerodynamic_provider().resistance_s_m()
        rb = DEPACGasQuasiLaminarResistance(
            friction_velocity_m_s=self.meteorology.friction_velocity_m_s,
            molecular_diffusivity_m2_s=self.species.molecular_diffusivity_m2_s,
            air_kinematic_viscosity_m2_s=self.meteorology.air_kinematic_viscosity_m2_s,
            prandtl_number=self.meteorology.prandtl_number,
        ).resistance_s_m()
        return ra, rb

    def _surface_paths(self) -> tuple[CompensationPath, ...]:
        paths: list[CompensationPath] = []
        gas_paths = self.surface.gas.paths
        if self.surface.gas.stomatal is not None:
            chi_s = self.compensation.concentration("stomatal")
            if chi_s is None:
                raise ValueError("active stomatal pathway requires explicit stomatal Gamma/temperature")
            r_stom = self.surface.gas.stomatal_resistance_s_m(
                environment=self.stomatal_environment,
                species_to_reference_diffusivity_ratio=self.species.stomatal_diffusivity_ratio,
            )
            if math.isfinite(r_stom):
                paths.append(
                    CompensationPath(
                        "stomatal",
                        r_stom,
                        chi_s,
                        self.compensation.provenance,
                    )
                )

        r_w = float(gas_paths.external_surface_resistance_s_m)
        if math.isfinite(r_w):
            chi_w = self.compensation.concentration("external")
            if chi_w is None:
                raise ValueError("active external pathway requires explicit external Gamma/temperature")
            paths.append(CompensationPath("external", r_w, chi_w, self.compensation.provenance))

        r_inc = float(gas_paths.in_canopy_resistance_s_m)
        r_soil = float(gas_paths.soil_resistance_s_m)
        if math.isinf(r_inc) or math.isinf(r_soil):
            r_soil_eff = math.inf
        else:
            r_soil_eff = r_inc + r_soil
        if math.isfinite(r_soil_eff):
            if r_soil_eff <= 0.0:
                raise ValueError("effective soil pathway resistance must be positive")
            chi_soil = self.compensation.concentration("soil")
            if chi_soil is None:
                raise ValueError("active soil pathway requires explicit soil Gamma/temperature")
            paths.append(
                CompensationPath("soil", r_soil_eff, chi_soil, self.compensation.provenance)
            )
        return tuple(paths)

    def result(self) -> BidirectionalExchangeResult:
        ra, rb = self._ra_rb()
        node = CanopyNodeExchange(
            atmospheric_concentration_ug_m3=self.atmospheric_concentration_ug_m3,
            ra_s_m=ra,
            rb_s_m=rb,
            pathways=self._surface_paths(),
        )
        chi_c, flux_up, pathway_fluxes, residual, regime = node.solve()
        return BidirectionalExchangeResult(
            species="NH3",
            atmospheric_concentration_ug_m3=float(self.atmospheric_concentration_ug_m3),
            canopy_concentration_ug_m3=chi_c,
            ra_s_m=ra,
            rb_s_m=rb,
            flux_upward_ug_m2_s=flux_up,
            regime=regime,
            pathway_fluxes=pathway_fluxes,
            mass_balance_residual_ug_m2_s=residual,
            species_provenance=self.species.provenance.compact_label,
            surface_provenance=self.surface.state.descriptor.provenance.compact_label,
            meteorology_provenance=self.meteorology.provenance.compact_label,
            compensation_provenance=self.compensation.provenance.compact_label,
        )

    def flux_upward_ug_m2_s(self) -> float:
        return self.result().flux_upward_ug_m2_s

    def downward_flux_ug_m2_s(self) -> float:
        return self.result().downward_flux_ug_m2_s
