"""Aerosol collection-efficiency kernels for GILTT-Py 2.0.

QA-030E isolates dimensionless surface-collection mechanisms used by the
Zhang et al. (2001) / Slinn dry-deposition lineage.  The Peters-Eiden
``(St/(alpha+St))**2`` impaction and interception formulas implemented by the
composite provider are the vegetated/rough-collector branch.  Smooth surfaces
use the separate Slinn-Slinn ``10**(-3/St)`` impaction branch and do not use the
collector-interception term in the same way.  Particle transport
properties are deliberately *not* derived here: Schmidt number, Stokes number,
and particle diameter are explicit inputs so Brownian diffusivity, settling,
slip correction, hygroscopic growth and the Stokes-number construction can be
audited independently in QA-030F.

For the Zhang-2001 algebraic collection model,

    E_B   = Sc**(-gamma),
    E_imp = (St / (alpha + St))**2,
    E_int = 0.5 * (d_p / A)**2,

where ``A`` is the characteristic collector radius.  ``gamma``, ``alpha`` and
``A`` are surface/season parameters in the original model and therefore remain
explicit.  No land-use lookup table and no rebound factor are hidden here.

The empirical expressions are evaluated literally without clipping to [0, 1].
That is intentional: numerical clipping would silently change the source
parameterization and obscure out-of-domain inputs.  Applicability of a surface
parameter set is a provenance/domain question, not a numerical saturation rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


ZHANG2001_MIN_BROWNIAN_EXPONENT = 0.5
ZHANG2001_MAX_BROWNIAN_EXPONENT = 2.0 / 3.0
ZHANG2001_IMPACTION_EXPONENT = 2.0
ZHANG2001_INTERCEPTION_COEFFICIENT = 0.5
ZHANG2001_INTERCEPTION_EXPONENT = 2.0


class AerosolCollectionEfficiencyProvider(Protocol):
    """Contract for a resolved set of aerosol collection efficiencies."""

    def efficiencies(self) -> "CollectionEfficiencies": ...


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def zhang2001_brownian_efficiency(*, schmidt_number: float, exponent: float) -> float:
    """Brownian-diffusion collection efficiency ``E_B = Sc**(-gamma)``.

    Zhang et al. (2001) describes ``gamma`` as surface dependent and generally
    between 1/2 and 2/3, following the Slinn lineage.  QA-030E enforces that
    source-family interval rather than accepting an undocumented exponent.
    """
    sc = _positive("schmidt_number", schmidt_number)
    gamma = _finite("exponent", exponent)
    if not ZHANG2001_MIN_BROWNIAN_EXPONENT <= gamma <= ZHANG2001_MAX_BROWNIAN_EXPONENT:
        raise ValueError("exponent must lie in the Zhang-2001 source-family interval [1/2, 2/3]")
    return sc ** (-gamma)


def zhang2001_impaction_efficiency(*, stokes_number: float, alpha: float) -> float:
    """Vegetated/rough-collector impaction ``E_imp=(St/(alpha+St))**2``.

    ``alpha`` is a source/surface parameter.  The exponent is fixed at two in
    the Zhang et al. (2001) implementation of the Peters-Eiden form.
    """
    st = _nonnegative("stokes_number", stokes_number)
    a = _positive("alpha", alpha)
    if st == 0.0:
        return 0.0
    ratio = st / (a + st)
    return ratio ** ZHANG2001_IMPACTION_EXPONENT



def zhang2001_smooth_impaction_efficiency(*, stokes_number: float) -> float:
    """Smooth-surface Slinn-Slinn/Zhang impaction ``E_imp=10**(-3/St)``.

    The exact limit as ``St -> 0+`` is zero, returned explicitly.  This branch
    is kept separate from the Peters-Eiden vegetated/rough-collector form.
    """
    st = _nonnegative("stokes_number", stokes_number)
    if st == 0.0:
        return 0.0
    return 10.0 ** (-3.0 / st)

def zhang2001_interception_efficiency(
    *, particle_diameter_m: float, collector_radius_m: float
) -> float:
    """Interception efficiency ``E_int = 0.5*(d_p/A)**2``.

    ``A`` is the characteristic collector radius in Zhang et al. (2001).  The
    formula is not clipped to one; callers must use a parameter set inside its
    documented physical applicability domain.
    """
    dp = _nonnegative("particle_diameter_m", particle_diameter_m)
    radius = _positive("collector_radius_m", collector_radius_m)
    ratio = dp / radius
    return ZHANG2001_INTERCEPTION_COEFFICIENT * ratio ** ZHANG2001_INTERCEPTION_EXPONENT


@dataclass(frozen=True)
class CollectionEfficiencies:
    """Dimensionless mechanism-resolved aerosol collection efficiencies."""

    brownian: float
    impaction: float
    interception: float

    def __post_init__(self) -> None:
        _nonnegative("brownian", self.brownian)
        _nonnegative("impaction", self.impaction)
        _nonnegative("interception", self.interception)

    @property
    def total(self) -> float:
        """Algebraic sum used by the Zhang-2001 surface-resistance family."""
        return math.fsum((self.brownian, self.impaction, self.interception))


@dataclass(frozen=True)
class Zhang2001SurfaceCollectionParameters:
    """Source-tagged parameters for the vegetated/rough-collector Z01 branch.

    QA-030E deliberately contains no land-use/season lookup table.  The caller
    must select ``gamma``, ``alpha`` and ``A`` from a documented surface source.
    """

    surface_label: str
    brownian_exponent: float
    impaction_alpha: float
    collector_radius_m: float
    provenance: str

    def __post_init__(self) -> None:
        if not str(self.surface_label).strip():
            raise ValueError("surface_label must be nonempty")
        # Reuse source-family validation with an arbitrary positive Sc.
        zhang2001_brownian_efficiency(schmidt_number=1.0, exponent=self.brownian_exponent)
        _positive("impaction_alpha", self.impaction_alpha)
        _positive("collector_radius_m", self.collector_radius_m)
        if not str(self.provenance).strip():
            raise ValueError("provenance must be nonempty")


@dataclass(frozen=True)
class AerosolCollectionState:
    """Dimensionless transport state plus particle diameter for QA-030E."""

    schmidt_number: float
    stokes_number: float
    particle_diameter_m: float

    def __post_init__(self) -> None:
        _positive("schmidt_number", self.schmidt_number)
        _nonnegative("stokes_number", self.stokes_number)
        _nonnegative("particle_diameter_m", self.particle_diameter_m)


@dataclass(frozen=True)
class Zhang2001CollectionEfficiencies:
    """Mechanism-resolved vegetated/rough-collector Z01 provider.

    Smooth-surface impaction uses :func:`zhang2001_smooth_impaction_efficiency`
    and interception is not represented by this composite contract.
    """

    state: AerosolCollectionState
    surface: Zhang2001SurfaceCollectionParameters

    @property
    def brownian_efficiency(self) -> float:
        return zhang2001_brownian_efficiency(
            schmidt_number=self.state.schmidt_number,
            exponent=self.surface.brownian_exponent,
        )

    @property
    def impaction_efficiency(self) -> float:
        return zhang2001_impaction_efficiency(
            stokes_number=self.state.stokes_number,
            alpha=self.surface.impaction_alpha,
        )

    @property
    def interception_efficiency(self) -> float:
        return zhang2001_interception_efficiency(
            particle_diameter_m=self.state.particle_diameter_m,
            collector_radius_m=self.surface.collector_radius_m,
        )

    def efficiencies(self) -> CollectionEfficiencies:
        return CollectionEfficiencies(
            brownian=self.brownian_efficiency,
            impaction=self.impaction_efficiency,
            interception=self.interception_efficiency,
        )
