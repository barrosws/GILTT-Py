"""QA-044 target-free exact transforms for inverse-Laplace robustness.

Families cover smooth decay, causal delay, damped oscillation, and a square-root
branch point representative of diffusion kernels.  These references are exact
and independent of GILTT transport calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import cmath
import math
from typing import Callable


@dataclass(frozen=True)
class ExactLaplaceCase:
    name: str
    laplace: Callable[[complex], complex]
    exact: Callable[[float], float]
    regime: str


def exponential_case(a: float = 0.7) -> ExactLaplaceCase:
    return ExactLaplaceCase(
        "exponential", lambda s: 1.0/(complex(s)+a), lambda t: math.exp(-a*t), "smooth_decay"
    )


def delayed_step_case(delay: float = 2.0) -> ExactLaplaceCase:
    return ExactLaplaceCase(
        "delayed_step", lambda s: cmath.exp(-delay*complex(s))/complex(s),
        lambda t: 0.0 if t < delay else 1.0, "causal_delay"
    )


def delayed_exponential_case(delay: float = 1.5, a: float = 0.4) -> ExactLaplaceCase:
    return ExactLaplaceCase(
        "delayed_exponential",
        lambda s: cmath.exp(-delay*complex(s))/(complex(s)+a),
        lambda t: 0.0 if t < delay else math.exp(-a*(t-delay)),
        "causal_delay_smooth_postfront",
    )


def damped_sine_case(a: float = 0.25, omega: float = 3.0) -> ExactLaplaceCase:
    return ExactLaplaceCase(
        "damped_sine",
        lambda s: omega/((complex(s)+a)**2+omega**2),
        lambda t: math.exp(-a*t)*math.sin(omega*t), "oscillatory"
    )


def damped_cosine_case(a: float = 0.2, omega: float = 4.0) -> ExactLaplaceCase:
    return ExactLaplaceCase(
        "damped_cosine",
        lambda s: (complex(s)+a)/((complex(s)+a)**2+omega**2),
        lambda t: math.exp(-a*t)*math.cos(omega*t), "oscillatory"
    )


def diffusion_erfc_case(b: float = 1.3) -> ExactLaplaceCase:
    # L^{-1}{exp(-b sqrt(s))/s} = erfc(b/(2 sqrt(t))).
    return ExactLaplaceCase(
        "diffusion_erfc",
        lambda s: cmath.exp(-b*cmath.sqrt(complex(s)))/complex(s),
        lambda t: math.erfc(b/(2.0*math.sqrt(t))), "branch_point_diffusion"
    )
