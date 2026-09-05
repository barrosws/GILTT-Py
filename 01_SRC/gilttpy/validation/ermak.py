"""Corrected Ermak (1977) semi-infinite deposition/settling benchmark.

This module is a validation reference, not a production closure.  It implements
Stockie's corrected Ermak solution for constant horizontal wind and constant
vertical eddy diffusivity on z >= 0, then analytically integrates the Gaussian
crosswind factor so the result can be compared directly with the 2-D
crosswind-integrated GILTT-Py transport equation.

Sign convention matches QA-038: z is positive upward, settling speed ``w_set``
is positive downward, and the positive-downward ground flux obeys

    (K dC/dz + w_set C)|_{z=0} = w_dep C(0).

For constant K and u, r = K x / u.  The corrected Ermak parameter is
``w0 = w_dep - 0.5*w_set``.  When both velocities vanish, the expression
reduces exactly to the reflected Gaussian half-space solution.

References
----------
Ermak, D. L. (1977), Atmospheric Environment 11, 231-237,
doi:10.1016/0004-6981(77)90140-8.
Stockie, J. M. (2011), SIAM Review 53, 349-372,
doi:10.1137/10080991X, Eq. (3.23), including correction of the original
Ermak typographical errors.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import erfcx

FloatArray = NDArray[np.float64]


def _positive(name: str, x: float) -> float:
    x = float(x)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return x


def _nonnegative(name: str, x: float) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def ermak_crosswind_integrated_concentration(
    *,
    x: float,
    z: ArrayLike,
    emission_rate: float,
    wind_speed: float,
    diffusivity: float,
    source_height: float,
    settling_velocity: float,
    deposition_velocity: float,
) -> FloatArray:
    """Return the corrected Ermak crosswind-integrated concentration.

    The benchmark is defined for x>0 and z>=0 on the semi-infinite vertical
    half-space. ``emission_rate`` is the line-source/crosswind-integrated source
    rate Q used by the 2-D equation.
    """
    x = _positive("x", x)
    q = _nonnegative("emission_rate", emission_rate)
    u = _positive("wind_speed", wind_speed)
    k = _positive("diffusivity", diffusivity)
    hsrc = _nonnegative("source_height", source_height)
    wset = _nonnegative("settling_velocity", settling_velocity)
    wdep = _nonnegative("deposition_velocity", deposition_velocity)
    zz = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(zz)) or np.any(zz < 0.0):
        raise ValueError("z must contain finite nonnegative heights")

    r = k*x/u
    sr = math.sqrt(r)
    w0 = wdep - 0.5*wset

    dm = zz - hsrc
    dp = zz + hsrc
    e_minus = np.exp(-(dm*dm)/(4.0*r))
    e_plus = np.exp(-(dp*dp)/(4.0*r))

    arg = dp/(2.0*sr) + w0*sr/k
    # Rewrite exp[w0(z+H)/K + w0^2 r/K^2] erfc(arg) as
    # exp[-(z+H)^2/(4r)] erfcx(arg), avoiding a large exp*small-erfc product.
    radiation = (2.0*w0*math.sqrt(math.pi*r)/k) * e_plus * erfcx(arg)
    bracket = e_minus + e_plus - radiation

    drift = np.exp(-wset*dm/(2.0*k) - (wset*wset*r)/(4.0*k*k))
    prefactor = q/(2.0*math.sqrt(math.pi*r)*u)
    out = prefactor * drift * bracket
    return np.asarray(out, dtype=np.float64)


def ermak_reflected_gaussian_limit(
    *, x: float, z: ArrayLike, emission_rate: float, wind_speed: float,
    diffusivity: float, source_height: float,
) -> FloatArray:
    """Closed reflected Gaussian half-space solution (w_set=w_dep=0)."""
    x = _positive("x", x)
    q = _nonnegative("emission_rate", emission_rate)
    u = _positive("wind_speed", wind_speed)
    k = _positive("diffusivity", diffusivity)
    hsrc = _nonnegative("source_height", source_height)
    zz = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(zz)) or np.any(zz < 0.0):
        raise ValueError("z must contain finite nonnegative heights")
    r = k*x/u
    return np.asarray(
        q/(2.0*math.sqrt(math.pi*r)*u) * (
            np.exp(-((zz-hsrc)**2)/(4.0*r))
            + np.exp(-((zz+hsrc)**2)/(4.0*r))
        ), dtype=np.float64,
    )


def ermak_ground_deposition_flux(**kwargs) -> float:
    """Return positive-downward Ermak ground deposition flux w_dep*C(x,0)."""
    wdep = _nonnegative("deposition_velocity", kwargs["deposition_velocity"])
    c0 = float(ermak_crosswind_integrated_concentration(z=np.asarray([0.0]), **kwargs)[0])
    return wdep*c0


def ermak_advective_flux_to_infinity(*, z_max: float | None = None, **kwargs) -> float:
    """Numerically integrate u*C over z>=0 for an independent mass-budget check."""
    u = _positive("wind_speed", kwargs["wind_speed"])
    x = _positive("x", kwargs["x"])
    k = _positive("diffusivity", kwargs["diffusivity"])
    hsrc = _nonnegative("source_height", kwargs["source_height"])
    if z_max is None:
        sigma = math.sqrt(2.0*k*x/u)
        z_max = max(hsrc + 14.0*sigma + 10.0, 14.0*sigma + 10.0)
    z_max = _positive("z_max", z_max)
    def f(z: float) -> float:
        return u*float(ermak_crosswind_integrated_concentration(z=np.asarray([z]), **kwargs)[0])
    val, _ = quad(f, 0.0, z_max, epsabs=1e-12, epsrel=2e-11, limit=300)
    return float(val)


def ermak_integrated_deposition(*, x_end: float, **kwargs) -> float:
    """Numerically integrate the exact Ermak surface flux from 0 to x_end."""
    x_end = _positive("x_end", x_end)
    base = dict(kwargs)
    base.pop("x", None)
    def f(x: float) -> float:
        # Avoid evaluating the singular source plane exactly; the x->0 flux is
        # exponentially suppressed for a source above the surface.
        if x == 0.0:
            return 0.0
        return ermak_ground_deposition_flux(x=x, **base)
    val, _ = quad(f, 0.0, x_end, epsabs=2e-11, epsrel=5e-10, limit=400, points=[x_end*1e-4, x_end*1e-2])
    return float(val)
