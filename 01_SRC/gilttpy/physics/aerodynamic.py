"""Aerodynamic-resistance providers for GILTT-Py 2.0.

QA-030A implements an explicit Monin-Obukhov similarity-theory (MOST)
scalar-transfer resistance.  It is intentionally not wired into the GILTT
lower-boundary condition yet: the PDE already resolves turbulent transport to
``z_lower``, so boundary coupling must separately rule out aerodynamic-transfer
double counting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

VON_KARMAN = 0.4


class AerodynamicResistanceProvider(Protocol):
    """Contract for a meteorology/surface-dependent aerodynamic resistance."""

    def resistance_s_m(self) -> float: ...


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def businger_dyer_phi_h(
    zeta: float,
    *,
    beta_h: float = 5.0,
    gamma_h: float = 16.0,
) -> float:
    """Dimensionless scalar-gradient function ``phi_h(z/L)``.

    The default Dyer/Businger-Dyer branch is

    * neutral: ``phi_h = 1``;
    * stable: ``phi_h = 1 + beta_h*zeta``;
    * unstable: ``phi_h = (1 - gamma_h*zeta)**(-1/2)``.

    ``beta_h`` and ``gamma_h`` are explicit so later land-surface or host-model
    conventions can be represented without silently changing the physics.
    """
    zeta = float(zeta)
    beta_h = _finite_positive("beta_h", beta_h)
    gamma_h = _finite_positive("gamma_h", gamma_h)
    if not math.isfinite(zeta):
        raise ValueError("zeta must be finite")
    if zeta == 0.0:
        return 1.0
    if zeta > 0.0:
        return 1.0 + beta_h * zeta
    return (1.0 - gamma_h * zeta) ** -0.5


def businger_dyer_psi_h(
    zeta: float,
    *,
    beta_h: float = 5.0,
    gamma_h: float = 16.0,
) -> float:
    """Integrated Businger-Dyer scalar stability correction ``psi_h``.

    The sign convention matches the common MOST transfer relation

    ``ln(z2/z1) - psi_h(z2/L) + psi_h(z1/L)``.
    """
    zeta = float(zeta)
    beta_h = _finite_positive("beta_h", beta_h)
    gamma_h = _finite_positive("gamma_h", gamma_h)
    if not math.isfinite(zeta):
        raise ValueError("zeta must be finite")
    if zeta == 0.0:
        return 0.0
    if zeta > 0.0:
        return -beta_h * zeta
    y = math.sqrt(1.0 - gamma_h * zeta)
    return 2.0 * math.log((1.0 + y) / 2.0)


def most_aerodynamic_resistance(
    *,
    friction_velocity_m_s: float,
    reference_height_m: float,
    scalar_roughness_length_m: float,
    displacement_height_m: float = 0.0,
    monin_obukhov_length_m: float = math.inf,
    von_karman: float = VON_KARMAN,
    beta_h: float = 5.0,
    gamma_h: float = 16.0,
) -> float:
    """Return scalar aerodynamic resistance in s m-1 from MOST.

    ``reference_height_m`` and ``displacement_height_m`` are heights above the
    material ground. ``scalar_roughness_length_m`` is the scalar-transfer
    roughness length measured above the displaced zero plane; it is deliberately
    *not* assumed equal to the momentum roughness length or to the GILTT resolved
    lower-interface height.

    Ra = [ln((z_ref-d)/z0h) - psi_h((z_ref-d)/L) + psi_h(z0h/L)]/(kappa*u*)

    A positive, finite friction velocity is required.  The exact calm limit is
    Ra -> infinity and is left to a later surface-state policy rather than being
    hidden by clipping ``u_*``.
    """
    ustar = _finite_positive("friction_velocity_m_s", friction_velocity_m_s)
    zref = _finite_positive("reference_height_m", reference_height_m)
    z0h = _finite_positive("scalar_roughness_length_m", scalar_roughness_length_m)
    d = _finite_nonnegative("displacement_height_m", displacement_height_m)
    kappa = _finite_positive("von_karman", von_karman)
    beta_h = _finite_positive("beta_h", beta_h)
    gamma_h = _finite_positive("gamma_h", gamma_h)

    z_eff = zref - d
    if z_eff <= 0.0:
        raise ValueError("reference_height_m must exceed displacement_height_m")
    if z_eff < z0h:
        raise ValueError(
            "effective reference height (reference_height_m - displacement_height_m) "
            "must be >= scalar_roughness_length_m"
        )

    L = float(monin_obukhov_length_m)
    if math.isnan(L) or L == 0.0:
        raise ValueError("monin_obukhov_length_m must be nonzero or +/-inf")

    log_term = math.log(z_eff / z0h)
    if math.isinf(L):
        stability_term = 0.0
    else:
        if not math.isfinite(L):
            # Only +/-inf are accepted as the exact neutral limit.
            raise ValueError("invalid monin_obukhov_length_m")
        psi_top = businger_dyer_psi_h(z_eff / L, beta_h=beta_h, gamma_h=gamma_h)
        psi_bottom = businger_dyer_psi_h(z0h / L, beta_h=beta_h, gamma_h=gamma_h)
        stability_term = -psi_top + psi_bottom

    bracket = log_term + stability_term
    # A zero bracket is the zero-thickness transfer limit.  Negative values
    # indicate an invalid geometry/convention combination for this closure.
    if bracket < -64.0 * math.ulp(max(1.0, abs(log_term))):
        raise ValueError("MOST transfer bracket became negative")
    return max(0.0, bracket) / (kappa * ustar)


@dataclass(frozen=True)
class MOSTAerodynamicResistance:
    """Typed Businger-Dyer MOST aerodynamic-resistance provider."""

    friction_velocity_m_s: float
    reference_height_m: float
    scalar_roughness_length_m: float
    displacement_height_m: float = 0.0
    monin_obukhov_length_m: float = math.inf
    von_karman: float = VON_KARMAN
    beta_h: float = 5.0
    gamma_h: float = 16.0

    def __post_init__(self) -> None:
        # Validate eagerly by evaluating the provider once.
        self.resistance_s_m()

    @property
    def effective_reference_height_m(self) -> float:
        return float(self.reference_height_m - self.displacement_height_m)

    @property
    def zeta_reference(self) -> float:
        L = float(self.monin_obukhov_length_m)
        return 0.0 if math.isinf(L) else self.effective_reference_height_m / L

    @property
    def zeta_roughness(self) -> float:
        L = float(self.monin_obukhov_length_m)
        return 0.0 if math.isinf(L) else float(self.scalar_roughness_length_m) / L

    def resistance_s_m(self) -> float:
        return most_aerodynamic_resistance(
            friction_velocity_m_s=self.friction_velocity_m_s,
            reference_height_m=self.reference_height_m,
            scalar_roughness_length_m=self.scalar_roughness_length_m,
            displacement_height_m=self.displacement_height_m,
            monin_obukhov_length_m=self.monin_obukhov_length_m,
            von_karman=self.von_karman,
            beta_h=self.beta_h,
            gamma_h=self.gamma_h,
        )
