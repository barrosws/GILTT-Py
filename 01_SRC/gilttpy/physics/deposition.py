"""Modern dry-deposition kernels for GILTT-Py 2.0.

This module is deliberately independent of historical reproduction code.
It provides small, testable physics kernels; land-use/canopy parameterization
is delegated to later modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import math

G = 9.80665


class DepositionModel(Protocol):
    """Contract for a lower-interface deposition model."""
    def deposition_velocity(self) -> float: ...
    def downward_flux(self, concentration: float) -> float: ...


def _require_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class ConstantVelocity:
    """Historical/control model: F_down = Vd * C at the resolved lower interface."""
    velocity_m_s: float

    def __post_init__(self):
        _require_nonnegative("velocity_m_s", self.velocity_m_s)

    def deposition_velocity(self) -> float:
        return float(self.velocity_m_s)

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity() * float(concentration)


@dataclass(frozen=True)
class GasResistance:
    """Unidirectional gas deposition by resistance-in-series model.

    Vd = 1 / (Ra + Rb + Rc).
    Bidirectional compensation-point exchange is intentionally deferred.

    QA-030C extends the kernel to accept ``+inf`` as a physically explicit closed
    surface pathway (for example Rc=+inf when every canopy uptake pathway is
    closed).  In that limit Vd=0 exactly.  NaN, negative values and -inf remain
    invalid.
    """
    ra_s_m: float
    rb_s_m: float
    rc_s_m: float

    def __post_init__(self):
        vals=[]
        for name, value in (("ra_s_m", self.ra_s_m), ("rb_s_m", self.rb_s_m), ("rc_s_m", self.rc_s_m)):
            value=float(value)
            if math.isnan(value) or value < 0.0 or value == -math.inf:
                raise ValueError(f"{name} must be nonnegative or +inf")
            vals.append(value)
        if all(v == 0.0 for v in vals):
            raise ValueError("total resistance must be positive")

    def deposition_velocity(self) -> float:
        total = self.ra_s_m + self.rb_s_m + self.rc_s_m
        return 0.0 if math.isinf(total) else 1.0 / total

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity() * float(concentration)


def cunningham_slip_correction(particle_diameter_m: float, mean_free_path_m: float = 6.6e-8) -> float:
    """Cunningham correction using the common 2.514/0.8/0.55 form."""
    dp=float(particle_diameter_m); lam=float(mean_free_path_m)
    if not math.isfinite(dp) or dp <= 0.0:
        raise ValueError("particle_diameter_m must be finite and positive")
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("mean_free_path_m must be finite and positive")
    return 1.0 + (lam/dp)*(2.514 + 0.8*math.exp(-0.55*dp/lam))


def stokes_settling_velocity(
    particle_diameter_m: float,
    particle_density_kg_m3: float,
    air_dynamic_viscosity_pa_s: float = 1.81e-5,
    mean_free_path_m: float = 6.6e-8,
) -> float:
    """Stokes gravitational settling with Cunningham slip correction."""
    dp=float(particle_diameter_m)
    rho=_require_nonnegative("particle_density_kg_m3", particle_density_kg_m3)
    mu=float(air_dynamic_viscosity_pa_s)
    if dp <= 0.0 or not math.isfinite(dp):
        raise ValueError("particle_diameter_m must be finite and positive")
    if rho <= 0.0:
        raise ValueError("particle_density_kg_m3 must be positive")
    if mu <= 0.0 or not math.isfinite(mu):
        raise ValueError("air_dynamic_viscosity_pa_s must be finite and positive")
    cc=cunningham_slip_correction(dp, mean_free_path_m)
    return G*rho*dp*dp*cc/(18.0*mu)


@dataclass(frozen=True)
class AerosolResistanceSettling:
    """Aerosol deposition combining gravitational settling and resistances.

    Uses the Venkatram-Pleim form:
        Vd = Vg / (1 - exp[-Vg (Ra + Rb)])
    evaluated stably with expm1. In the Vg -> 0 limit this tends to
    1/(Ra+Rb).
    """
    ra_s_m: float
    rb_s_m: float
    settling_velocity_m_s: float

    def __post_init__(self):
        ra=_require_nonnegative("ra_s_m",self.ra_s_m)
        rb=_require_nonnegative("rb_s_m",self.rb_s_m)
        _require_nonnegative("settling_velocity_m_s",self.settling_velocity_m_s)
        if ra+rb <= 0.0:
            raise ValueError("Ra + Rb must be positive")

    def deposition_velocity(self) -> float:
        r=self.ra_s_m+self.rb_s_m
        vg=float(self.settling_velocity_m_s)
        if vg == 0.0:
            return 1.0/r
        denom=-math.expm1(-vg*r)
        return vg/denom

    def downward_flux(self, concentration: float) -> float:
        return self.deposition_velocity()*float(concentration)


@dataclass(frozen=True)
class ResolvedLowerInterface:
    """Metadata binding deposition to the resolved turbulent-domain interface.

    z_lower is not asserted to be the material ground surface.
    """
    z_lower_m: float
    model: DepositionModel

    def __post_init__(self):
        _require_nonnegative("z_lower_m", self.z_lower_m)

    def downward_flux(self, concentration_at_z_lower: float) -> float:
        return self.model.downward_flux(concentration_at_z_lower)
