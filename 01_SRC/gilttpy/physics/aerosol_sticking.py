"""Aerosol rebound/sticking and explicit surface-state kernels for GILTT-Py 2.0.

QA-030G isolates the Zhang et al. (2001) / Slinn empirical rebound correction
from collection efficiencies and particle-transport physics.

For a dry surface, Zhang et al. states that particles larger than 5 micrometres
may rebound and uses the sticking fraction

    R1 = exp(-sqrt(St)).

For wet surfaces no particle rebound is assumed, so R1 = 1.  The 5 micrometre
activation diameter is exposed as a parameter rather than hidden.  Surface
wetness is also explicit: this module does not infer wetness from ambient RH,
precipitation, dew point, vegetation class, or particle hygroscopicity.

The empirical Zhang/Slinn sticking fraction is a reference parameterization,
not a universal collision-mechanics law.  Mechanistic adhesion/restitution,
material dependence, surface roughness and liquid-film physics remain outside
QA-030G and are held for the integrated aerosol-scheme audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M = 5.0e-6


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_or_posinf(name: str, value: float) -> float:
    value = float(value)
    if math.isnan(value) or value < 0.0 or value == -math.inf:
        raise ValueError(f"{name} must be nonnegative or +inf")
    return value


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def zhang2001_dry_sticking_fraction(*, stokes_number: float) -> float:
    """Return the Zhang/Slinn dry-surface sticking fraction ``exp(-sqrt(St))``.

    ``St=+inf`` is accepted as the exact asymptotic limit and returns zero.
    Applicability to a particle size is handled separately by
    :func:`zhang2001_sticking_fraction`.
    """
    st = _nonnegative_or_posinf("stokes_number", stokes_number)
    if st == math.inf:
        return 0.0
    return math.exp(-math.sqrt(st))


@dataclass(frozen=True)
class AerosolSurfaceState:
    """Explicit surface wetness state with mandatory provenance.

    ``is_wet`` is deliberately supplied by the caller.  QA-030G contains no
    humidity/dew/precipitation threshold that silently changes the state.
    """

    surface_label: str
    is_wet: bool
    provenance: str

    def __post_init__(self) -> None:
        if not str(self.surface_label).strip():
            raise ValueError("surface_label must be nonempty")
        if type(self.is_wet) is not bool:
            raise ValueError("is_wet must be an explicit bool")
        if not str(self.provenance).strip():
            raise ValueError("provenance must be nonempty")


def zhang2001_sticking_fraction(
    *,
    stokes_number: float,
    particle_diameter_m: float,
    surface_is_wet: bool,
    rebound_activation_diameter_m: float = ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M,
) -> float:
    """Return the source-scoped Zhang-2001 sticking fraction ``R1``.

    Rules are explicit and ordered:

    1. Wet surface -> R1 = 1 for every particle size.
    2. Dry surface and dp <= activation diameter -> R1 = 1.
    3. Dry surface and dp > activation diameter -> exp(-sqrt(St)).

    Zhang et al. (2001) motivates rebound for particles larger than 5 micrometres.
    The strict ``>`` criterion is preserved by the default threshold.  No smooth
    interpolation is invented at the threshold.
    """
    st = _nonnegative_or_posinf("stokes_number", stokes_number)
    dp = _finite_positive("particle_diameter_m", particle_diameter_m)
    threshold = _finite_positive(
        "rebound_activation_diameter_m", rebound_activation_diameter_m
    )
    if type(surface_is_wet) is not bool:
        raise ValueError("surface_is_wet must be an explicit bool")
    if surface_is_wet or dp <= threshold:
        return 1.0
    return zhang2001_dry_sticking_fraction(stokes_number=st)


def sticking_adjusted_collection_efficiency(
    *, total_collection_efficiency: float, sticking_fraction: float
) -> float:
    """Apply a sticking fraction to an already resolved collection efficiency.

    This helper performs only the multiplicative Zhang-2001 operation
    ``E_effective = E_total * R1``.  It does not construct a surface resistance or
    deposition velocity; those integrations remain for QA-030H/QA-034.
    """
    e = float(total_collection_efficiency)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError("total_collection_efficiency must be finite and nonnegative")
    r1 = _unit_interval("sticking_fraction", sticking_fraction)
    return e * r1


@dataclass(frozen=True)
class Zhang2001ReboundSticking:
    """Typed, provenance-carrying Zhang-2001 rebound/sticking provider."""

    surface: AerosolSurfaceState
    rebound_activation_diameter_m: float = ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M
    provenance: str = "Zhang2001_Slinn1982_empirical_rebound"

    def __post_init__(self) -> None:
        _finite_positive(
            "rebound_activation_diameter_m", self.rebound_activation_diameter_m
        )
        if not str(self.provenance).strip():
            raise ValueError("provenance must be nonempty")

    def sticking_fraction(self, *, stokes_number: float, particle_diameter_m: float) -> float:
        return zhang2001_sticking_fraction(
            stokes_number=stokes_number,
            particle_diameter_m=particle_diameter_m,
            surface_is_wet=self.surface.is_wet,
            rebound_activation_diameter_m=self.rebound_activation_diameter_m,
        )
