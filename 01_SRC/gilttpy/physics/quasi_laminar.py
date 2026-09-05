"""Gas quasi-laminar resistance for GILTT-Py 2.0.

QA-030B implements a gas-side quasi-laminar resistance provider following the
Wesely/DEPAC resistance formulation

    Rb = 2/(kappa*u*) * (Sc/Pr)**(2/3),
    Sc = nu_air / D_species,air.

The module keeps transport-property provenance explicit.  Molecular
diffusivities are not silently inferred from molecular mass.  A small reference
library reproduces the 0 degC, 1 atm air-diffusivity values summarized by
Massman (1998); entries based on model estimates rather than direct/combined
data are labelled accordingly.  The Massman temperature/pressure scaling used
here is the table-level D(T,p)=D0*(T/T0)**1.81*(p0/p) relation.

Like QA-030A aerodynamic resistance, Rb is intentionally *not* wired into the
GILTT lower-boundary condition yet.  Coupling Ra, Rb and later Rc to the
resolved z_lower interface is a separate physics-partition gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

from .aerodynamic import VON_KARMAN

STANDARD_TEMPERATURE_K = 273.15
STANDARD_PRESSURE_PA = 101_325.0
AIR_SPECIFIC_GAS_CONSTANT_J_KG_K = 287.05
SUTHERLAND_C1_PA_S_K_HALF = 1.458e-6
SUTHERLAND_S_K = 110.4
DEFAULT_AIR_PRANDTL = 0.72
MASSMAN_DEFAULT_TEMPERATURE_EXPONENT = 1.81


class QuasiLaminarResistanceProvider(Protocol):
    """Contract for a gas/particle quasi-laminar resistance provider."""

    def resistance_s_m(self) -> float: ...


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def schmidt_number(
    *,
    air_kinematic_viscosity_m2_s: float,
    molecular_diffusivity_m2_s: float,
) -> float:
    """Return the gas Schmidt number ``Sc = nu_air / D_m``."""
    nu = _finite_positive("air_kinematic_viscosity_m2_s", air_kinematic_viscosity_m2_s)
    dmol = _finite_positive("molecular_diffusivity_m2_s", molecular_diffusivity_m2_s)
    return nu / dmol


def depac_gas_quasi_laminar_resistance(
    *,
    friction_velocity_m_s: float,
    molecular_diffusivity_m2_s: float,
    air_kinematic_viscosity_m2_s: float,
    prandtl_number: float = DEFAULT_AIR_PRANDTL,
    von_karman: float = VON_KARMAN,
) -> float:
    """Return gas quasi-laminar resistance in s m-1.

    The implemented source family is the Wesely/DEPAC gas relation

        Rb = 2/(kappa*u*) * (Sc/Pr)**(2/3).

    No lower bound is imposed on ``u*`` and no Schmidt/Prandtl clipping is
    performed.  The exact calm limit therefore remains Rb -> infinity rather
    than being hidden by a numerical floor.
    """
    ustar = _finite_positive("friction_velocity_m_s", friction_velocity_m_s)
    pr = _finite_positive("prandtl_number", prandtl_number)
    kappa = _finite_positive("von_karman", von_karman)
    sc = schmidt_number(
        air_kinematic_viscosity_m2_s=air_kinematic_viscosity_m2_s,
        molecular_diffusivity_m2_s=molecular_diffusivity_m2_s,
    )
    return 2.0 / (kappa * ustar) * (sc / pr) ** (2.0 / 3.0)


def air_dynamic_viscosity_sutherland_pa_s(temperature_k: float) -> float:
    """Dynamic viscosity of dry air from Sutherland's law.

    Uses ``mu = 1.458e-6*T**1.5/(T+110.4)`` in SI units, a standard NASA air
    form appropriate to ordinary atmospheric temperatures.
    """
    t = _finite_positive("temperature_k", temperature_k)
    return SUTHERLAND_C1_PA_S_K_HALF * t ** 1.5 / (t + SUTHERLAND_S_K)


def dry_air_density_ideal_kg_m3(
    *,
    temperature_k: float,
    pressure_pa: float,
    specific_gas_constant_j_kg_k: float = AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
) -> float:
    """Dry-air density from the ideal-gas relation ``rho = p/(R*T)``."""
    t = _finite_positive("temperature_k", temperature_k)
    p = _finite_positive("pressure_pa", pressure_pa)
    r = _finite_positive("specific_gas_constant_j_kg_k", specific_gas_constant_j_kg_k)
    return p / (r * t)


def air_kinematic_viscosity_sutherland_m2_s(
    *,
    temperature_k: float,
    pressure_pa: float,
) -> float:
    """Dry-air kinematic viscosity ``nu = mu/rho`` using Sutherland + ideal gas."""
    mu = air_dynamic_viscosity_sutherland_pa_s(temperature_k)
    rho = dry_air_density_ideal_kg_m3(temperature_k=temperature_k, pressure_pa=pressure_pa)
    return mu / rho


@dataclass(frozen=True)
class GasDiffusivityReference:
    """Reference molecular diffusivity in air at 273.15 K and 1 atm."""

    species: str
    d0_m2_s: float
    provenance: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.species.strip():
            raise ValueError("species must be nonempty")
        _finite_positive("d0_m2_s", self.d0_m2_s)
        if not self.provenance.strip():
            raise ValueError("provenance must be nonempty")


# Massman (1998), Table 2, values in cm2/s converted to m2/s by 1e-4.
# For O3, NO and NO2 the data column is NM/IM, so the selected values below
# are explicitly the CO62 model estimates from the same table rather than
# measurements.  That distinction is part of the API provenance.
MASSMAN_1998_AIR_DIFFUSIVITY: Mapping[str, GasDiffusivityReference] = {
    "H2O": GasDiffusivityReference("H2O", 0.2178e-4, "Massman1998_data_reanalysis"),
    "CO2": GasDiffusivityReference("CO2", 0.1381e-4, "Massman1998_data_reanalysis"),
    "CH4": GasDiffusivityReference("CH4", 0.1952e-4, "Massman1998_data_reanalysis"),
    "CO": GasDiffusivityReference("CO", 0.1807e-4, "Massman1998_combined_binary_data"),
    "SO2": GasDiffusivityReference("SO2", 0.1089e-4, "Massman1998_combined_air_N2_data"),
    "O3": GasDiffusivityReference(
        "O3", 0.1444e-4, "Massman1998_CO62_model_estimate",
        "Massman Table 2 marks O3/air as never measured in the data column.",
    ),
    "NH3": GasDiffusivityReference("NH3", 0.1978e-4, "Massman1998_Wintergerst_regression"),
    "N2O": GasDiffusivityReference("N2O", 0.1436e-4, "Massman1998_data_reanalysis"),
    "NO": GasDiffusivityReference(
        "NO", 0.1988e-4, "Massman1998_CO62_model_estimate",
        "Massman Table 2 marks NO/air as never measured in the data column.",
    ),
    "NO2": GasDiffusivityReference(
        "NO2", 0.1361e-4, "Massman1998_CO62_model_estimate",
        "Massman Table 2 marks NO2/air as impossible to measure in the data column.",
    ),
}


def massman_1998_air_diffusivity_m2_s(
    species: str,
    *,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    pressure_pa: float = STANDARD_PRESSURE_PA,
    temperature_exponent: float = MASSMAN_DEFAULT_TEMPERATURE_EXPONENT,
) -> float:
    """Return species diffusivity in air using the Massman-1998 table baseline.

    The table-level scaling is

        D(T,p) = D0 * (T/273.15 K)**1.81 * (101325 Pa/p).

    ``temperature_exponent`` is exposed to avoid hiding an alternative future
    species-specific transport-property convention.
    """
    key = str(species).strip().upper()
    if key not in MASSMAN_1998_AIR_DIFFUSIVITY:
        valid = ", ".join(sorted(MASSMAN_1998_AIR_DIFFUSIVITY))
        raise KeyError(f"unsupported Massman-1998 species {species!r}; choose one of: {valid}")
    t = _finite_positive("temperature_k", temperature_k)
    p = _finite_positive("pressure_pa", pressure_pa)
    exponent = _finite_positive("temperature_exponent", temperature_exponent)
    ref = MASSMAN_1998_AIR_DIFFUSIVITY[key]
    return ref.d0_m2_s * (t / STANDARD_TEMPERATURE_K) ** exponent * (STANDARD_PRESSURE_PA / p)


@dataclass(frozen=True)
class DEPACGasQuasiLaminarResistance:
    """Typed Rb provider for explicit externally supplied transport properties."""

    friction_velocity_m_s: float
    molecular_diffusivity_m2_s: float
    air_kinematic_viscosity_m2_s: float
    prandtl_number: float = DEFAULT_AIR_PRANDTL
    von_karman: float = VON_KARMAN

    def __post_init__(self) -> None:
        self.resistance_s_m()

    @property
    def schmidt_number(self) -> float:
        return schmidt_number(
            air_kinematic_viscosity_m2_s=self.air_kinematic_viscosity_m2_s,
            molecular_diffusivity_m2_s=self.molecular_diffusivity_m2_s,
        )

    def resistance_s_m(self) -> float:
        return depac_gas_quasi_laminar_resistance(
            friction_velocity_m_s=self.friction_velocity_m_s,
            molecular_diffusivity_m2_s=self.molecular_diffusivity_m2_s,
            air_kinematic_viscosity_m2_s=self.air_kinematic_viscosity_m2_s,
            prandtl_number=self.prandtl_number,
            von_karman=self.von_karman,
        )


@dataclass(frozen=True)
class MassmanDEPACGasQuasiLaminarResistance:
    """Convenience provider with explicit Massman-1998 + Sutherland provenance.

    This is a traceable reference implementation for QA and sensitivity work,
    not a claim that one property database is optimal for every future species.
    """

    species: str
    friction_velocity_m_s: float
    temperature_k: float = 298.15
    pressure_pa: float = STANDARD_PRESSURE_PA
    prandtl_number: float = DEFAULT_AIR_PRANDTL
    temperature_exponent: float = MASSMAN_DEFAULT_TEMPERATURE_EXPONENT
    von_karman: float = VON_KARMAN

    def __post_init__(self) -> None:
        self.resistance_s_m()

    @property
    def diffusivity_reference(self) -> GasDiffusivityReference:
        key = str(self.species).strip().upper()
        if key not in MASSMAN_1998_AIR_DIFFUSIVITY:
            # Reuse the canonical validation/error message.
            massman_1998_air_diffusivity_m2_s(key)
        return MASSMAN_1998_AIR_DIFFUSIVITY[key]

    @property
    def molecular_diffusivity_m2_s(self) -> float:
        return massman_1998_air_diffusivity_m2_s(
            self.species,
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_pa,
            temperature_exponent=self.temperature_exponent,
        )

    @property
    def air_kinematic_viscosity_m2_s(self) -> float:
        return air_kinematic_viscosity_sutherland_m2_s(
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_pa,
        )

    @property
    def schmidt_number(self) -> float:
        return schmidt_number(
            air_kinematic_viscosity_m2_s=self.air_kinematic_viscosity_m2_s,
            molecular_diffusivity_m2_s=self.molecular_diffusivity_m2_s,
        )

    def resistance_s_m(self) -> float:
        return depac_gas_quasi_laminar_resistance(
            friction_velocity_m_s=self.friction_velocity_m_s,
            molecular_diffusivity_m2_s=self.molecular_diffusivity_m2_s,
            air_kinematic_viscosity_m2_s=self.air_kinematic_viscosity_m2_s,
            prandtl_number=self.prandtl_number,
            von_karman=self.von_karman,
        )
