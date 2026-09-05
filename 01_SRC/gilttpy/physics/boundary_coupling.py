"""Resolved-interface dry-deposition flux partition for GILTT-Py 2.0.

QA-036 closes the *physics-side* aerodynamic-resistance partition required
before dry-deposition providers can be mapped to the explicit GILTT lower
interface ``z_lower``.  It deliberately does not modify a spectral transport
operator; typed operator activation, including affine bidirectional forcing,
is assigned to QA-037.

The key distinction is between

* turbulent transfer already resolved by the PDE above ``z_lower``; and
* transfer that remains parameterized below ``z_lower``.

For a scalar reference height ``z_ref``, lower resolved interface ``z_lower``,
zero-plane displacement ``d`` and scalar roughness length ``z0h``, the MOST
resistance is additive by endpoints::

    Ra(z_ref -> d+z0h)
      = Ra(z_ref -> z_lower) + Ra(z_lower -> d+z0h).

The first segment is *not* inserted into the GILTT boundary.  Only the residual
sub-interface segment is combined with quasi-laminar and surface resistances.
For unidirectional gas uptake::

    J_down = k_int * C(z_lower)
    k_int  = 1 / (Ra_sub + Rb + Rc).

For a bidirectional surface network, the surface pathways can be reduced to an
equivalent resistance ``Rc_eq`` and conductance-weighted compensation
concentration ``C_eq``.  The resolved-interface law is then affine::

    J_down = k_int * [C(z_lower) - C_eq]
    k_int  = 1 / (Ra_sub + Rb + Rc_eq).

Positive ``J_down`` means removal from the resolved atmospheric domain;
negative ``J_down`` means emission into it.

All complete aerosol closures frozen in QA-034 include gravitational settling
``Vg``.  They are intentionally *not* mapped to the boundary here because the
same settling contribution is scheduled to enter the transport operator in
QA-038.  That partition remains a QA-038/QA-039 mass-conservation gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .aerodynamic import VON_KARMAN, businger_dyer_psi_h
from .bidirectional_exchange import CompensationPath
from .deposition import GasResistance
from .gas_deposition import (
    GasDepositionMeteorology,
    GasSpeciesDepositionProperties,
    StandaloneUnidirectionalGasDeposition,
)
from .stomatal import StomatalEnvironment
from .surface_abstraction import SurfacePhysicsBundle

FloatArray = NDArray[np.float64]

AEROSOL_BOUNDARY_COUPLING_STATUS = (
    "HOLD_COMPLETE_AEROSOL_CLOSURES_INCLUDE_VG__AWAIT_QA038_QA039_CONSERVATIVE_PARTITION"
)
SPECTRAL_OPERATOR_COUPLING_STATUS = (
    "PHYSICS_FLUX_PARTITION_VERIFIED__HOLD_TYPED_SPECTRAL_OPERATOR_ACTIVATION_TO_QA037"
)


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


def _resistance_nonnegative_or_inf(name: str, value: float) -> float:
    value = float(value)
    if math.isnan(value) or value < 0.0 or value == -math.inf:
        raise ValueError(f"{name} must be nonnegative or +inf")
    return value


def most_scalar_transfer_resistance_between_heights(
    *,
    friction_velocity_m_s: float,
    upper_height_m: float,
    lower_height_m: float,
    displacement_height_m: float = 0.0,
    monin_obukhov_length_m: float = math.inf,
    von_karman: float = VON_KARMAN,
    beta_h: float = 5.0,
    gamma_h: float = 16.0,
) -> float:
    """MOST scalar-transfer resistance between two absolute heights, s m-1.

    Both heights are above the same material-ground datum.  The effective MOST
    coordinates are ``z-d``.  Equal endpoints give the exact zero-thickness
    limit.  The function does not assume that either endpoint is a momentum or
    scalar roughness length.
    """
    ustar = _finite_positive("friction_velocity_m_s", friction_velocity_m_s)
    zu = _finite_positive("upper_height_m", upper_height_m)
    zl = _finite_positive("lower_height_m", lower_height_m)
    d = _finite_nonnegative("displacement_height_m", displacement_height_m)
    kappa = _finite_positive("von_karman", von_karman)
    beta = _finite_positive("beta_h", beta_h)
    gamma = _finite_positive("gamma_h", gamma_h)
    if zu < zl:
        raise ValueError("upper_height_m must be >= lower_height_m")
    zu_eff = zu - d
    zl_eff = zl - d
    if zu_eff <= 0.0 or zl_eff <= 0.0:
        raise ValueError("both transfer endpoints must exceed displacement_height_m")
    if zu == zl:
        return 0.0

    L = float(monin_obukhov_length_m)
    if math.isnan(L) or L == 0.0:
        raise ValueError("monin_obukhov_length_m must be nonzero or +/-inf")
    log_term = math.log(zu_eff / zl_eff)
    if math.isinf(L):
        stability = 0.0
    else:
        if not math.isfinite(L):
            raise ValueError("invalid monin_obukhov_length_m")
        stability = -businger_dyer_psi_h(zu_eff / L, beta_h=beta, gamma_h=gamma)
        stability += businger_dyer_psi_h(zl_eff / L, beta_h=beta, gamma_h=gamma)
    bracket = log_term + stability
    tol = 64.0 * math.ulp(max(1.0, abs(log_term), abs(stability)))
    if bracket < -tol:
        raise ValueError("MOST transfer bracket became negative")
    return max(0.0, bracket) / (kappa * ustar)


@dataclass(frozen=True)
class AerodynamicResistancePartition:
    """Endpoint-explicit MOST partition around the resolved lower interface."""

    friction_velocity_m_s: float
    reference_height_m: float
    interface_height_m: float
    scalar_roughness_length_m: float
    displacement_height_m: float = 0.0
    monin_obukhov_length_m: float = math.inf
    von_karman: float = VON_KARMAN
    beta_h: float = 5.0
    gamma_h: float = 16.0

    def __post_init__(self) -> None:
        _finite_positive("friction_velocity_m_s", self.friction_velocity_m_s)
        _finite_positive("reference_height_m", self.reference_height_m)
        _finite_positive("interface_height_m", self.interface_height_m)
        _finite_positive("scalar_roughness_length_m", self.scalar_roughness_length_m)
        _finite_nonnegative("displacement_height_m", self.displacement_height_m)
        endpoint = self.surface_transfer_endpoint_height_m
        if self.reference_height_m < self.interface_height_m:
            raise ValueError("reference_height_m must be >= interface_height_m")
        if self.interface_height_m < endpoint:
            raise ValueError(
                "interface_height_m must be at or above displacement_height_m + scalar_roughness_length_m"
            )
        # Evaluate all segments eagerly to validate stability/geometry.
        _ = self.full_reference_to_surface_s_m
        _ = self.upper_most_segment_s_m
        _ = self.residual_subinterface_s_m

    @property
    def surface_transfer_endpoint_height_m(self) -> float:
        return float(self.displacement_height_m + self.scalar_roughness_length_m)

    def _segment(self, upper: float, lower: float) -> float:
        return most_scalar_transfer_resistance_between_heights(
            friction_velocity_m_s=self.friction_velocity_m_s,
            upper_height_m=upper,
            lower_height_m=lower,
            displacement_height_m=self.displacement_height_m,
            monin_obukhov_length_m=self.monin_obukhov_length_m,
            von_karman=self.von_karman,
            beta_h=self.beta_h,
            gamma_h=self.gamma_h,
        )

    @property
    def full_reference_to_surface_s_m(self) -> float:
        return self._segment(self.reference_height_m, self.surface_transfer_endpoint_height_m)

    @property
    def upper_most_segment_s_m(self) -> float:
        """MOST-equivalent reference-to-interface segment; diagnostic only."""
        return self._segment(self.reference_height_m, self.interface_height_m)

    @property
    def residual_subinterface_s_m(self) -> float:
        """Only aerodynamic segment eligible for the resolved-interface closure."""
        return self._segment(self.interface_height_m, self.surface_transfer_endpoint_height_m)

    @property
    def additivity_residual_s_m(self) -> float:
        return self.full_reference_to_surface_s_m - (
            self.upper_most_segment_s_m + self.residual_subinterface_s_m
        )


@dataclass(frozen=True)
class LinearResolvedInterfaceFluxLaw:
    """Affine Robin law ``J_down = k*(C_interface - C_eq)``."""

    sink_velocity_m_s: float
    equilibrium_concentration: float = 0.0
    label: str = "resolved-interface linear flux"

    def __post_init__(self) -> None:
        k = _finite_nonnegative("sink_velocity_m_s", self.sink_velocity_m_s)
        ceq = float(self.equilibrium_concentration)
        if not math.isfinite(ceq) or ceq < 0.0:
            raise ValueError("equilibrium_concentration must be finite and nonnegative")
        if k == 0.0 and ceq != 0.0:
            raise ValueError("zero sink velocity requires zero equilibrium concentration")
        if not str(self.label).strip():
            raise ValueError("label must be nonempty")

    def downward_flux(self, concentration_at_interface: float) -> float:
        c = float(concentration_at_interface)
        if not math.isfinite(c) or c < 0.0:
            raise ValueError("concentration_at_interface must be finite and nonnegative")
        return float(self.sink_velocity_m_s) * (c - float(self.equilibrium_concentration))

    def weak_terms(self, lower_basis_values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return ``(B, f)`` for ``M Y' + S Y + B Y = f``.

        If ``b`` evaluates the basis at ``z_lower``, then
        ``B=k b b^T`` and ``f=k*C_eq*b``.  Hence ``B@y-f`` is exactly
        ``b*J_down`` when ``C(z_lower)=b@y``.
        """
        b = np.asarray(lower_basis_values, dtype=np.float64)
        if b.ndim != 1 or b.size < 1 or not np.all(np.isfinite(b)):
            raise ValueError("lower_basis_values must be a finite one-dimensional vector")
        k = float(self.sink_velocity_m_s)
        B = k * np.outer(b, b)
        f = k * float(self.equilibrium_concentration) * b
        return np.asarray(B, dtype=np.float64), np.asarray(f, dtype=np.float64)


@dataclass(frozen=True)
class ResolvedInterfaceGasBoundaryResult:
    species: str
    interface_height_m: float
    full_ra_s_m: float
    upper_most_segment_s_m: float
    residual_ra_s_m: float
    rb_s_m: float
    rc_s_m: float
    interface_sink_velocity_m_s: float
    reference_deposition_velocity_m_s: float
    partition_additivity_residual_s_m: float
    coupling_status: str = SPECTRAL_OPERATOR_COUPLING_STATUS

    @property
    def interface_law(self) -> LinearResolvedInterfaceFluxLaw:
        return LinearResolvedInterfaceFluxLaw(
            self.interface_sink_velocity_m_s, 0.0, f"{self.species} unidirectional resolved-interface"
        )

    @property
    def residual_boundary_resistance_s_m(self) -> float:
        if math.isinf(self.rc_s_m):
            return math.inf
        return self.residual_ra_s_m + self.rb_s_m + self.rc_s_m

    @property
    def reference_total_resistance_s_m(self) -> float:
        if math.isinf(self.rc_s_m):
            return math.inf
        return self.full_ra_s_m + self.rb_s_m + self.rc_s_m

    def reconstructed_reference_velocity_m_s(self) -> float:
        """Reinsert the upper MOST segment in series as an algebraic audit."""
        r_boundary = self.residual_boundary_resistance_s_m
        if math.isinf(r_boundary):
            return 0.0
        total = self.upper_most_segment_s_m + r_boundary
        return 0.0 if math.isinf(total) else 1.0 / total


@dataclass(frozen=True)
class ResolvedInterfaceUnidirectionalGasBoundary:
    """Map the verified QA-032 gas chain to a resolved GILTT interface."""

    species: GasSpeciesDepositionProperties
    surface: SurfacePhysicsBundle
    meteorology: GasDepositionMeteorology
    interface_height_m: float
    stomatal_environment: StomatalEnvironment | None = None

    def partition(self) -> AerodynamicResistancePartition:
        m = self.meteorology
        return AerodynamicResistancePartition(
            friction_velocity_m_s=m.friction_velocity_m_s,
            reference_height_m=m.reference_height_m,
            interface_height_m=self.interface_height_m,
            scalar_roughness_length_m=m.scalar_roughness_length_m,
            displacement_height_m=m.displacement_height_m,
            monin_obukhov_length_m=m.monin_obukhov_length_m,
        )

    def result(self) -> ResolvedInterfaceGasBoundaryResult:
        standalone = StandaloneUnidirectionalGasDeposition(
            self.species, self.surface, self.meteorology, self.stomatal_environment
        ).result()
        part = self.partition()
        scale = max(1.0, abs(standalone.ra_s_m), abs(part.full_reference_to_surface_s_m))
        if abs(standalone.ra_s_m - part.full_reference_to_surface_s_m) > 256.0 * math.ulp(scale):
            raise ArithmeticError("QA-032 Ra and QA-036 endpoint partition are inconsistent")
        k = GasResistance(part.residual_subinterface_s_m, standalone.rb_s_m, standalone.rc_s_m).deposition_velocity()
        return ResolvedInterfaceGasBoundaryResult(
            species=standalone.species,
            interface_height_m=float(self.interface_height_m),
            full_ra_s_m=standalone.ra_s_m,
            upper_most_segment_s_m=part.upper_most_segment_s_m,
            residual_ra_s_m=part.residual_subinterface_s_m,
            rb_s_m=standalone.rb_s_m,
            rc_s_m=standalone.rc_s_m,
            interface_sink_velocity_m_s=k,
            reference_deposition_velocity_m_s=standalone.deposition_velocity_m_s,
            partition_additivity_residual_s_m=part.additivity_residual_s_m,
        )


@dataclass(frozen=True)
class ResolvedInterfaceBidirectionalGasBoundary:
    """Reduce QA-033-style surface reservoirs to an affine interface law."""

    partition: AerodynamicResistancePartition
    rb_s_m: float
    pathways: tuple[CompensationPath, ...]

    def __post_init__(self) -> None:
        _resistance_nonnegative_or_inf("rb_s_m", self.rb_s_m)
        if not isinstance(self.pathways, tuple):
            object.__setattr__(self, "pathways", tuple(self.pathways))
        labels = [p.label for p in self.pathways]
        if len(labels) != len(set(labels)):
            raise ValueError("pathway labels must be unique")

    @property
    def open_pathways(self) -> tuple[CompensationPath, ...]:
        return tuple(p for p in self.pathways if p.is_open)

    @property
    def equivalent_surface_resistance_s_m(self) -> float:
        paths = self.open_pathways
        if not paths:
            return math.inf
        g = math.fsum(p.conductance_m_s for p in paths)
        return math.inf if g == 0.0 else 1.0 / g

    @property
    def equilibrium_concentration(self) -> float:
        paths = self.open_pathways
        if not paths:
            return 0.0
        g = math.fsum(p.conductance_m_s for p in paths)
        if g == 0.0:
            return 0.0
        return math.fsum(
            p.conductance_m_s * p.compensation_concentration_ug_m3 for p in paths
        ) / g

    def interface_law(self) -> LinearResolvedInterfaceFluxLaw:
        rc = self.equivalent_surface_resistance_s_m
        rb = float(self.rb_s_m)
        if math.isinf(rc) or math.isinf(rb):
            return LinearResolvedInterfaceFluxLaw(0.0, 0.0, "bidirectional isolated interface")
        total = self.partition.residual_subinterface_s_m + rb + rc
        k = 1.0 / total
        return LinearResolvedInterfaceFluxLaw(k, self.equilibrium_concentration, "bidirectional resolved-interface")

    def reconstructed_reference_downward_flux(self, reference_concentration: float) -> float:
        """Algebraic two-stage audit against the full reference-height network."""
        c_ref = float(reference_concentration)
        if not math.isfinite(c_ref) or c_ref < 0.0:
            raise ValueError("reference_concentration must be finite and nonnegative")
        law = self.interface_law()
        if law.sink_velocity_m_s == 0.0:
            return 0.0
        r_upper = self.partition.upper_most_segment_s_m
        k = law.sink_velocity_m_s
        ceq = law.equilibrium_concentration
        c_int = (c_ref + r_upper * k * ceq) / (1.0 + r_upper * k)
        return law.downward_flux(c_int)


def parallel_surface_equivalent(paths: Iterable[CompensationPath]) -> tuple[float, float]:
    """Return ``(Rc_eq, C_eq)`` for a set of QA-033 compensation pathways."""
    obj = tuple(paths)
    # A zero-residual aerodynamic partition is sufficient because this helper
    # only needs the surface reduction; use direct conductance algebra here.
    open_paths = tuple(p for p in obj if p.is_open)
    if not open_paths:
        return math.inf, 0.0
    g = math.fsum(p.conductance_m_s for p in open_paths)
    if g == 0.0:
        return math.inf, 0.0
    rc = 1.0 / g
    ceq = math.fsum(p.conductance_m_s * p.compensation_concentration_ug_m3 for p in open_paths) / g
    return rc, ceq
