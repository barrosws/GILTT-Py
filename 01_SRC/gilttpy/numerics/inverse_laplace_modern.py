"""Modern robust numerical inverse-Laplace providers.

QA-043 deliberately keeps the historical fixed-Talbot implementation intact
in ``gilttpy.solvers.transient_2d``.  Fixed Talbot is fast for smooth transforms
but is not robust for delayed/Heaviside fronts because its deformed contour
extends into the left half-plane.  The modern path therefore exposes a de Hoog
provider through mpmath, while Laplace-space model evaluations remain the
verified complex-double GILTT-Py operators.

The implementation is numerical infrastructure only.  It contains no
observational calibration, transport parameter, or historical target.
"""
from __future__ import annotations

from typing import Callable
import math

import numpy as np


def dehoog_inverse_laplace(
    laplace_fn: Callable[[complex], complex],
    t: float,
    *,
    degree: int = 28,
    working_dps: int = 40,
) -> float:
    """Invert a scalar Laplace transform with the de Hoog algorithm.

    ``laplace_fn`` is evaluated in complex128 arithmetic by the surrounding
    GILTT-Py solver.  mpmath supplies the robust Bromwich/Fourier inversion
    arithmetic; precision above complex128 therefore stabilizes the inversion
    summation but does not claim more than double-precision model evaluations.
    """
    t = float(t)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and positive")
    if int(degree) != degree or int(degree) < 8:
        raise ValueError("degree must be an integer >= 8")
    if int(working_dps) != working_dps or int(working_dps) < 20:
        raise ValueError("working_dps must be an integer >= 20")

    degree = int(degree)
    working_dps = int(working_dps)

    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise ImportError("de Hoog inversion requires the optional mpmath dependency") from exc

    def wrapped(p):
        value = complex(laplace_fn(complex(p)))
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise FloatingPointError("nonfinite Laplace-space model evaluation")
        return mp.mpc(value.real, value.imag)

    with mp.workdps(working_dps):
        value = mp.invertlaplace(
            wrapped,
            mp.mpf(t),
            method="dehoog",
            degree=degree,
        )
        out = float(mp.re(value))
    if not math.isfinite(out):
        raise FloatingPointError("nonfinite de Hoog inverse-Laplace result")
    return out


def dehoog_consensus_inverse_laplace(
    laplace_fn: Callable[[complex], complex],
    t: float,
    *,
    degrees: tuple[int, ...] = (24, 26, 28),
    working_dps: int = 40,
) -> float:
    """Target-free degree consensus for complex128 Laplace-model evaluations.

    The same complex128 Laplace model is inverted independently at an odd,
    strictly increasing set of de Hoog degrees and the median estimate is
    returned. This does not claim precision beyond complex128. It limits
    sensitivity to degree-dependent amplification of low-level model-evaluation
    roundoff observed in the QA043 portability campaign.
    """
    raw = tuple(degrees)
    if len(raw) < 3 or len(raw) % 2 == 0:
        raise ValueError("degrees must contain an odd number of at least three values")

    parsed: list[int] = []
    for degree in raw:
        try:
            value = int(degree)
        except (TypeError, ValueError) as exc:
            raise ValueError("degrees must contain integers") from exc
        if value != degree:
            raise ValueError("degrees must contain integers")
        parsed.append(value)

    degrees = tuple(parsed)
    if tuple(sorted(degrees)) != degrees or len(set(degrees)) != len(degrees):
        raise ValueError("degrees must be strictly increasing and unique")
    if degrees[0] < 8 or degrees[-1] > 30:
        raise ValueError("consensus degrees must lie in [8,30] for complex128 model evaluations")
    if int(working_dps) != working_dps or int(working_dps) < 20:
        raise ValueError("working_dps must be integer >= 20")

    values = np.asarray(
        [
            dehoog_inverse_laplace(
                laplace_fn,
                t,
                degree=degree,
                working_dps=int(working_dps),
            )
            for degree in degrees
        ],
        dtype=np.float64,
    )
    out = float(np.median(values))
    if not math.isfinite(out):
        raise FloatingPointError("nonfinite de Hoog consensus inverse-Laplace result")
    return out
