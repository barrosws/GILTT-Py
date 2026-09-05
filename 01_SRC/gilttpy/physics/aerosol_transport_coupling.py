"""Resolved-interface aerosol flux partition with gravitational settling.

QA-038 moves gravitational settling into the *resolved transport operator*.
The lower-boundary law must therefore be interpreted as the **total downward
flux leaving the resolved domain**, not as an extra sink added on top of an
already-counted settling loss.

With upward vertical coordinate ``z`` and positive settling speed ``Vg``, the
resolved vertical downward flux is

    F_down = Kz * dC/dz + Vg * C.

For a resistance-like unresolved layer of total resistance ``R`` below the
resolved interface, the Venkatram--Pleim mass-consistent closure is

    k_VP = Vg / (1 - exp(-Vg R)),

with the exact ``Vg -> 0`` limit ``1/R``.  This closure is compositional: a
resolved resistance segment followed by an unresolved segment gives exactly the
same effective flux as one combined resistance when settling is represented in
both segments through the conservative flux law.

The Zhang-2001/Slinn additive family is retained as a distinct local split:

    k_Z01,interface = Vg + 1/(Ra_sub + Rs).

This is a valid local total-flux boundary law, and mirrors operational systems
that treat gravitational settling separately from the non-settling ZH01 dry
capture term.  It is **not** claimed to reconstruct the original
reference-height additive formula after a nonzero resolved settling segment;
that lack of partition invariance is explicit model-form provenance.

Historical branches are not imported or modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .deposition import AerosolResistanceSettling

FloatArray = NDArray[np.float64]


SETTLING_PARTITION_STATUS = (
    "RESOLVED_SETTLING_OPERATOR__TOTAL_INTERFACE_FLUX_EXPLICIT__QA039_GLOBAL_MASS_AUDIT_PENDING"
)


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def effective_velocity_above_resolved_resistance(
    *,
    settling_velocity_m_s: float,
    lower_total_exit_velocity_m_s: float,
    resolved_resistance_s_m: float,
) -> float:
    """Propagate a lower total-flux law through a resolved settling layer.

    For constant downward flux ``J`` and resistance coordinate ``R`` satisfying

        J = Vg*C + dC/dR,

    the effective upper-level velocity after a resolved resistance ``Rr`` is

        1/k_upper = 1/Vg + (1/k_lower - 1/Vg) exp(-Vg Rr).

    The exact zero-settling limit is the usual series-resistance identity

        1/k_upper = Rr + 1/k_lower.

    ``k_lower=0`` represents zero removal and returns zero. ``+inf`` is allowed
    as the perfect-sink lower limit.
    """
    vg = _nonnegative("settling_velocity_m_s", settling_velocity_m_s)
    rr = _nonnegative("resolved_resistance_s_m", resolved_resistance_s_m)
    kl = float(lower_total_exit_velocity_m_s)
    if math.isnan(kl) or kl < 0.0 or kl == -math.inf:
        raise ValueError("lower_total_exit_velocity_m_s must be nonnegative or +inf")
    if kl == 0.0:
        return 0.0
    inv_kl = 0.0 if math.isinf(kl) else 1.0 / kl
    if vg == 0.0:
        denom = rr + inv_kl
        return math.inf if denom == 0.0 else 1.0 / denom
    inv_upper = 1.0 / vg + (inv_kl - 1.0 / vg) * math.exp(-vg * rr)
    if inv_upper <= 0.0 or not math.isfinite(inv_upper):
        raise FloatingPointError("invalid composed settling resistance state")
    return 1.0 / inv_upper


@dataclass(frozen=True)
class ResolvedSettlingAerosolFluxLaw:
    """Homogeneous aerosol law for the total downward flux at ``z_lower``.

    ``sink_velocity_m_s`` is the total velocity multiplying interface
    concentration in the conservative boundary flux.  ``settling_velocity_m_s``
    is carried separately so audits can prove which part is represented by the
    volume drift operator and which part is additional surface capture.
    """

    settling_velocity_m_s: float
    sink_velocity_m_s: float
    model_family: str
    provenance: str
    label: str
    equilibrium_concentration: float = 0.0
    coupling_status: str = SETTLING_PARTITION_STATUS

    def __post_init__(self) -> None:
        vg = _nonnegative("settling_velocity_m_s", self.settling_velocity_m_s)
        k = _nonnegative("sink_velocity_m_s", self.sink_velocity_m_s)
        if k + 64.0 * math.ulp(max(1.0, vg, k)) < vg:
            raise ValueError(
                "QA-038 resolved-interface aerosol sink must be at least the downward settling speed"
            )
        if float(self.equilibrium_concentration) != 0.0:
            raise ValueError("QA-038 aerosol settling law requires zero equilibrium concentration")
        if not str(self.model_family).strip():
            raise ValueError("model_family must be nonempty")
        if not str(self.provenance).strip():
            raise ValueError("provenance must be nonempty")
        if not str(self.label).strip():
            raise ValueError("label must be nonempty")

    @property
    def nonsettling_increment_m_s(self) -> float:
        return max(0.0, float(self.sink_velocity_m_s) - float(self.settling_velocity_m_s))

    def downward_flux(self, concentration_at_interface: float) -> float:
        c = _nonnegative("concentration_at_interface", concentration_at_interface)
        return float(self.sink_velocity_m_s) * c

    def weak_terms(self, lower_basis_values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        b = np.asarray(lower_basis_values, dtype=np.float64)
        if b.ndim != 1 or b.size < 1 or not np.all(np.isfinite(b)):
            raise ValueError("lower_basis_values must be a finite one-dimensional vector")
        B = float(self.sink_velocity_m_s) * np.outer(b, b)
        f = np.zeros_like(b)
        return np.asarray(B, dtype=np.float64), np.asarray(f, dtype=np.float64)


def venkatram_pleim_resolved_interface_flux_law(
    *,
    settling_velocity_m_s: float,
    unresolved_resistance_s_m: float,
    provenance: str,
    label: str = "VP1999 resolved-interface total aerosol flux",
) -> ResolvedSettlingAerosolFluxLaw:
    """Mass-consistent VP total-flux law for the unresolved sub-interface layer."""
    vg = _nonnegative("settling_velocity_m_s", settling_velocity_m_s)
    r = _positive("unresolved_resistance_s_m", unresolved_resistance_s_m)
    k = AerosolResistanceSettling(0.0, r, vg).deposition_velocity()
    return ResolvedSettlingAerosolFluxLaw(
        settling_velocity_m_s=vg,
        sink_velocity_m_s=k,
        model_family="venkatram_pleim_1999_mass_consistent",
        provenance=provenance,
        label=label,
    )


def zhang2001_split_resolved_interface_flux_law(
    *,
    settling_velocity_m_s: float,
    residual_aerodynamic_resistance_s_m: float,
    surface_resistance_s_m: float,
    provenance: str,
    label: str = "Zhang2001 split settling + local nonsettling capture",
) -> ResolvedSettlingAerosolFluxLaw:
    """Local Z01 split with settling in transport and capture below interface.

    The total interface velocity is ``Vg + 1/(Ra_sub+Rs)``.  ``Rs=+inf`` gives
    the exact pure-settling limit.  This function does not claim reference-
    height invariance of the original additive Zhang formula.
    """
    vg = _nonnegative("settling_velocity_m_s", settling_velocity_m_s)
    ra = _nonnegative("residual_aerodynamic_resistance_s_m", residual_aerodynamic_resistance_s_m)
    rs = float(surface_resistance_s_m)
    if math.isnan(rs) or rs < 0.0 or rs == -math.inf:
        raise ValueError("surface_resistance_s_m must be nonnegative or +inf")
    if math.isinf(rs):
        capture = 0.0
    else:
        total = ra + rs
        if total <= 0.0:
            raise ValueError("Ra_sub + Rs must be positive for finite surface resistance")
        capture = 1.0 / total
    return ResolvedSettlingAerosolFluxLaw(
        settling_velocity_m_s=vg,
        sink_velocity_m_s=vg + capture,
        model_family="zhang2001_slinn_split_resolved_settling",
        provenance=provenance,
        label=label,
    )
