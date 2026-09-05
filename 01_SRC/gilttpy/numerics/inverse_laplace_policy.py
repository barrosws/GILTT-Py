"""QA-044 explicit modern inverse-Laplace policy.

The modern GILTT-Py path uses de Hoog, Knight & Stokes inversion by default.
Historical Fixed Talbot remains available only through the historical solver and
is never selected silently by this policy.  The policy is numerical
infrastructure: no observational target or transport calibration appears here.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from gilttpy.numerics.inverse_laplace_modern import dehoog_inverse_laplace


class InverseLaplaceMethod(str, Enum):
    DEHOOG = "dehoog"


@dataclass(frozen=True)
class ModernInverseLaplacePolicy:
    method: InverseLaplaceMethod = InverseLaplaceMethod.DEHOOG
    degree: int = 24
    working_dps: int = 40
    label: str = "QA044 modern de Hoog default"
    provenance: str = "de Hoog, Knight & Stokes (1982); QA044 target-free robustness campaign"

    def __post_init__(self) -> None:
        if self.method is not InverseLaplaceMethod.DEHOOG:
            raise ValueError("modern policy currently permits only explicit de Hoog inversion")
        if int(self.degree) != self.degree or not (8 <= int(self.degree) <= 30):
            raise ValueError("degree must be an integer in [8,30] for complex128 model evaluations")
        if int(self.working_dps) != self.working_dps or int(self.working_dps) < 20:
            raise ValueError("working_dps must be integer >= 20")
        if not self.label.strip() or not self.provenance.strip():
            raise ValueError("label and provenance are required")

    def invert(self, laplace_fn: Callable[[complex], complex], t: float) -> float:
        return dehoog_inverse_laplace(
            laplace_fn, t, degree=int(self.degree), working_dps=int(self.working_dps)
        )
