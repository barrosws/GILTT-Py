"""Gas canopy/surface resistance architecture for GILTT-Py 2.0.

QA-030C closes only the *pathway-combination* layer for gas surface resistance.
It does not parameterize stomatal physiology; QA-030D will supply stomatal
resistance from radiation, temperature, VPD, soil-water, LAI and phenology.

The source-frozen DEPAC structure is

    Rc = (1/Rstom + 1/Rsoil_eff + 1/Rw)**(-1),
    Rsoil_eff = Rinc + Rsoil.

An explicit-path provider additionally permits a mesophyll resistance in series
with stomatal entry, reflecting the separation used in several contemporary
schemes.  Setting Rmes=0 recovers the DEPAC stomatal branch exactly.

Positive infinity is a first-class "closed pathway" value.  Therefore a canopy
with all uptake pathways closed has Rc=+inf rather than an arbitrary large
finite cap.  A zero-resistance pathway gives Rc=0 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class SurfaceResistanceProvider(Protocol):
    """Contract for a gas surface/canopy resistance provider."""

    def resistance_s_m(self) -> float: ...


def _resistance_value(name: str, value: float) -> float:
    """Validate a nonnegative resistance, allowing +inf as a closed pathway."""
    value = float(value)
    if math.isnan(value) or value < 0.0 or value == -math.inf:
        raise ValueError(f"{name} must be nonnegative or +inf")
    return value


def series_resistance_s_m(*resistances_s_m: float) -> float:
    """Equivalent resistance for components in series.

    +inf in any component closes the series path.  At least one component is
    required so accidental empty-path construction cannot silently return zero.
    """
    if not resistances_s_m:
        raise ValueError("at least one series resistance is required")
    values = [_resistance_value(f"resistance[{i}]", r) for i, r in enumerate(resistances_s_m)]
    if any(math.isinf(r) for r in values):
        return math.inf
    return math.fsum(values)


def parallel_resistance_s_m(*resistances_s_m: float) -> float:
    """Equivalent resistance for independent uptake pathways in parallel.

    A zero-resistance path dominates and gives zero.  +inf contributes zero
    conductance.  If all paths are closed (+inf), the result is +inf.
    """
    if not resistances_s_m:
        raise ValueError("at least one parallel resistance is required")
    values = [_resistance_value(f"resistance[{i}]", r) for i, r in enumerate(resistances_s_m)]
    if any(r == 0.0 for r in values):
        return 0.0
    conductances = [1.0 / r for r in values if not math.isinf(r)]
    if not conductances:
        return math.inf
    total_conductance = math.fsum(conductances)
    if math.isinf(total_conductance):
        return 0.0
    return 1.0 / total_conductance


@dataclass(frozen=True)
class DEPACCanopyResistance:
    """Exact three-path DEPAC canopy resistance combiner.

    Parameters are bulk pathway resistances in s m-1:

    - ``stomatal_resistance_s_m`` = Rstom;
    - ``external_surface_resistance_s_m`` = Rw;
    - ``in_canopy_resistance_s_m`` = Rinc;
    - ``soil_resistance_s_m`` = Rsoil.

    The provider computes Rsoil_eff=Rinc+Rsoil and then combines Rstom,
    Rsoil_eff and Rw in parallel.  No meteorological, land-use, physiology or
    species parameterization is hidden here.
    """

    stomatal_resistance_s_m: float
    external_surface_resistance_s_m: float
    in_canopy_resistance_s_m: float
    soil_resistance_s_m: float

    def __post_init__(self) -> None:
        _resistance_value("stomatal_resistance_s_m", self.stomatal_resistance_s_m)
        _resistance_value("external_surface_resistance_s_m", self.external_surface_resistance_s_m)
        _resistance_value("in_canopy_resistance_s_m", self.in_canopy_resistance_s_m)
        _resistance_value("soil_resistance_s_m", self.soil_resistance_s_m)

    @property
    def effective_soil_resistance_s_m(self) -> float:
        return series_resistance_s_m(self.in_canopy_resistance_s_m, self.soil_resistance_s_m)

    def resistance_s_m(self) -> float:
        return parallel_resistance_s_m(
            self.stomatal_resistance_s_m,
            self.effective_soil_resistance_s_m,
            self.external_surface_resistance_s_m,
        )

    @property
    def conductance_m_s(self) -> float:
        rc = self.resistance_s_m()
        if math.isinf(rc):
            return 0.0
        if rc == 0.0:
            return math.inf
        return 1.0 / rc


@dataclass(frozen=True)
class ExplicitPathCanopyResistance:
    """Transparent canopy resistance with optional mesophyll separation.

    The three independent branches are

        stomatal: Rstom + Rmes,
        external: Rw,
        soil:     Rinc + Rsoil,

    followed by parallel conductance addition.  This is an architecture layer,
    not a species or physiology parameterization.  ``mesophyll_resistance_s_m``
    defaults to zero so DEPAC-like stomatal input is represented without
    modification when desired.
    """

    stomatal_resistance_s_m: float
    external_surface_resistance_s_m: float
    in_canopy_resistance_s_m: float
    soil_resistance_s_m: float
    mesophyll_resistance_s_m: float = 0.0

    def __post_init__(self) -> None:
        _resistance_value("stomatal_resistance_s_m", self.stomatal_resistance_s_m)
        _resistance_value("external_surface_resistance_s_m", self.external_surface_resistance_s_m)
        _resistance_value("in_canopy_resistance_s_m", self.in_canopy_resistance_s_m)
        _resistance_value("soil_resistance_s_m", self.soil_resistance_s_m)
        _resistance_value("mesophyll_resistance_s_m", self.mesophyll_resistance_s_m)

    @property
    def stomatal_path_resistance_s_m(self) -> float:
        return series_resistance_s_m(self.stomatal_resistance_s_m, self.mesophyll_resistance_s_m)

    @property
    def effective_soil_resistance_s_m(self) -> float:
        return series_resistance_s_m(self.in_canopy_resistance_s_m, self.soil_resistance_s_m)

    def resistance_s_m(self) -> float:
        return parallel_resistance_s_m(
            self.stomatal_path_resistance_s_m,
            self.external_surface_resistance_s_m,
            self.effective_soil_resistance_s_m,
        )

    @property
    def conductance_m_s(self) -> float:
        rc = self.resistance_s_m()
        if math.isinf(rc):
            return 0.0
        if rc == 0.0:
            return math.inf
        return 1.0 / rc
