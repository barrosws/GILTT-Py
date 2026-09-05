"""Lognormal aerosol size distributions and size-integrated dry deposition.

QA-035 lifts the complete monodisperse QA-034 aerosol closures to explicitly
polydisperse, lognormal aerosol populations. It does not alter either complete
closure and it does not construct a GILTT lower-boundary condition.

For a number-lognormal diameter distribution, X=ln(D) is Gaussian with
mu=ln(Dg) and s=ln(sigma_g). The normalized k-th diameter moment is

    E[D**k] = Dg**k * exp(0.5*k**2*s**2).

For spherical particles with constant density within a mode, mass weighting
multiplies the number distribution by D**3. Completing the square gives another
Gaussian in ln(D), with the same width and shifted mean mu_M=mu+3*s**2. Thus

    Dg_mass = Dg_number * exp(3*ln(sigma_g)**2).

The current particle-physics kernel is not asserted valid on the infinite upper
tail of a mathematical lognormal distribution. Therefore integration requires
an explicit, provenance-carrying diameter domain [Dmin,Dmax]. The code reports
retained number and mass fractions and computes conditional in-domain effective
velocities. It never silently interprets an unmodelled tail as deposited.

Numerical integration uses Gauss-Legendre quadrature in ln(D) on the explicit
domain. For Zhang-2001 dry surfaces the integral is split exactly at the
source-scoped rebound activation diameter when that point lies inside the
integration domain, avoiding loss of convergence across the discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss

from .aerosol_deposition import (
    AerosolDepositionMeteorology,
    AerosolDepositionModelFamily,
    StandaloneVenkatramPleimAerosolDeposition,
    StandaloneZhang2001AerosolDeposition,
    VenkatramPleimNonsettlingResistance,
)
from .particle_physics import AerosolParticleProperties
from .surface_abstraction import ProvenanceRecord, SurfacePhysicsBundle


class AerosolDistributionWeighting(str, Enum):
    NUMBER = "number"
    MASS = "mass"


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


@dataclass(frozen=True)
class AerosolDiameterDomain:
    """Explicit physical/numerical diameter applicability domain."""

    min_diameter_m: float
    max_diameter_m: float
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        dmin = _positive("min_diameter_m", self.min_diameter_m)
        dmax = _positive("max_diameter_m", self.max_diameter_m)
        if dmax <= dmin:
            raise ValueError("max_diameter_m must exceed min_diameter_m")


@dataclass(frozen=True)
class LognormalAerosolMode:
    """One number-lognormal spherical-particle mode with explicit provenance."""

    label: str
    total_number_concentration_m3: float
    geometric_mean_diameter_m: float
    geometric_std_dev: float
    density_kg_m3: float
    diameter_basis: str
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("label must be nonempty")
        _nonnegative("total_number_concentration_m3", self.total_number_concentration_m3)
        _positive("geometric_mean_diameter_m", self.geometric_mean_diameter_m)
        sg = float(self.geometric_std_dev)
        if not math.isfinite(sg) or sg < 1.0:
            raise ValueError("geometric_std_dev must be finite and >= 1")
        _positive("density_kg_m3", self.density_kg_m3)
        if not str(self.diameter_basis).strip():
            raise ValueError("diameter_basis must be nonempty")

    @property
    def ln_sigma(self) -> float:
        return math.log(float(self.geometric_std_dev))

    def normalized_moment_m_power(self, order: float) -> float:
        k = float(order)
        if not math.isfinite(k):
            raise ValueError("moment order must be finite")
        dg = float(self.geometric_mean_diameter_m)
        s = self.ln_sigma
        return dg**k * math.exp(0.5 * k * k * s * s)

    def total_moment_m_power_per_m3(self, order: float) -> float:
        return float(self.total_number_concentration_m3) * self.normalized_moment_m_power(order)

    @property
    def total_mass_concentration_kg_m3(self) -> float:
        return (
            float(self.total_number_concentration_m3)
            * float(self.density_kg_m3)
            * math.pi / 6.0
            * self.normalized_moment_m_power(3.0)
        )

    @property
    def mass_geometric_mean_diameter_m(self) -> float:
        s = self.ln_sigma
        return float(self.geometric_mean_diameter_m) * math.exp(3.0 * s * s)

    def _weighted_mu(self, weighting: AerosolDistributionWeighting) -> float:
        mu = math.log(float(self.geometric_mean_diameter_m))
        if weighting is AerosolDistributionWeighting.MASS:
            return mu + 3.0 * self.ln_sigma**2
        if weighting is AerosolDistributionWeighting.NUMBER:
            return mu
        raise ValueError("unsupported aerosol distribution weighting")

    def retained_fraction(self, *, domain: AerosolDiameterDomain, weighting: AerosolDistributionWeighting) -> float:
        if self.geometric_std_dev == 1.0:
            d = self.geometric_mean_diameter_m
            return 1.0 if domain.min_diameter_m <= d <= domain.max_diameter_m else 0.0
        mu = self._weighted_mu(weighting)
        s = self.ln_sigma
        za = (math.log(domain.min_diameter_m) - mu) / s
        zb = (math.log(domain.max_diameter_m) - mu) / s
        return max(0.0, min(1.0, _normal_cdf(zb) - _normal_cdf(za)))


@dataclass(frozen=True)
class AerosolSizeDistribution:
    modes: tuple[LognormalAerosolMode, ...]
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("at least one aerosol mode is required")
        object.__setattr__(self, "modes", modes)
        labels = [m.label for m in modes]
        if len(labels) != len(set(labels)):
            raise ValueError("aerosol mode labels must be unique")
        bases = {str(m.diameter_basis).strip() for m in modes}
        if len(bases) != 1:
            raise ValueError("all modes in one distribution must share the same diameter_basis")
        if self.total_number_concentration_m3 <= 0.0:
            raise ValueError("distribution total number concentration must be positive")

    @property
    def diameter_basis(self) -> str:
        return self.modes[0].diameter_basis

    @property
    def total_number_concentration_m3(self) -> float:
        return math.fsum(m.total_number_concentration_m3 for m in self.modes)

    @property
    def total_mass_concentration_kg_m3(self) -> float:
        return math.fsum(m.total_mass_concentration_kg_m3 for m in self.modes)


class VenkatramPleimSizeResistanceProvider(Protocol):
    def resistance(self, *, particle: AerosolParticleProperties, meteorology: AerosolDepositionMeteorology) -> VenkatramPleimNonsettlingResistance:
        ...


@dataclass(frozen=True)
class ConstantVenkatramPleimSizeResistance:
    """Explicit constant-over-size VP resistance for controlled QA/reference use."""

    resistance_s_m: float
    surface_label: str
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        _nonnegative("resistance_s_m", self.resistance_s_m)
        if not str(self.surface_label).strip():
            raise ValueError("surface_label must be nonempty")

    def resistance(self, *, particle: AerosolParticleProperties, meteorology: AerosolDepositionMeteorology) -> VenkatramPleimNonsettlingResistance:
        del particle, meteorology
        return VenkatramPleimNonsettlingResistance(self.resistance_s_m, self.surface_label, self.provenance)


@dataclass(frozen=True)
class ModeIntegratedDepositionResult:
    label: str
    total_number_concentration_m3: float
    total_mass_concentration_kg_m3: float
    retained_number_fraction: float
    retained_mass_fraction: float
    in_domain_number_concentration_m3: float
    in_domain_mass_concentration_kg_m3: float
    number_weighted_vd_m_s: float
    mass_weighted_vd_m_s: float
    geometric_mean_diameter_m: float
    mass_geometric_mean_diameter_m: float
    geometric_std_dev: float
    density_kg_m3: float
    diameter_basis: str
    provenance: str


@dataclass(frozen=True)
class AerosolSizeDistributionDepositionResult:
    model_family: AerosolDepositionModelFamily
    diameter_basis: str
    domain_min_diameter_m: float
    domain_max_diameter_m: float
    retained_number_fraction: float
    retained_mass_fraction: float
    number_weighted_vd_m_s: float
    mass_weighted_vd_m_s: float
    total_number_concentration_m3: float
    total_mass_concentration_kg_m3: float
    in_domain_number_concentration_m3: float
    in_domain_mass_concentration_kg_m3: float
    mode_results: tuple[ModeIntegratedDepositionResult, ...]
    quadrature_order: int
    distribution_provenance: str
    domain_provenance: str

    def effective_deposition_velocity_m_s(self, weighting: AerosolDistributionWeighting) -> float:
        if weighting is AerosolDistributionWeighting.NUMBER:
            return self.number_weighted_vd_m_s
        if weighting is AerosolDistributionWeighting.MASS:
            return self.mass_weighted_vd_m_s
        raise ValueError("unsupported aerosol distribution weighting")

    def downward_flux_for_in_domain_concentration(self, *, concentration: float, weighting: AerosolDistributionWeighting) -> float:
        """Flux for a concentration already restricted to the reported diameter domain."""
        return _nonnegative("concentration", concentration) * self.effective_deposition_velocity_m_s(weighting)


@dataclass(frozen=True)
class StandaloneAerosolSizeDistributionDeposition:
    distribution: AerosolSizeDistribution
    diameter_domain: AerosolDiameterDomain
    model_family: AerosolDepositionModelFamily
    meteorology: AerosolDepositionMeteorology
    surface: SurfacePhysicsBundle | None = None
    vp_resistance_provider: VenkatramPleimSizeResistanceProvider | None = None
    quadrature_order: int = 64

    def __post_init__(self) -> None:
        q = int(self.quadrature_order)
        if q != self.quadrature_order or q < 8 or q > 256:
            raise ValueError("quadrature_order must be an integer in [8, 256]")
        if self.model_family is AerosolDepositionModelFamily.ZHANG2001_SLINN:
            if self.surface is None:
                raise ValueError("Zhang-2001 size integration requires an explicit SurfacePhysicsBundle")
            if self.vp_resistance_provider is not None:
                raise ValueError("VP resistance provider cannot be supplied to Zhang-2001 integration")
        elif self.model_family is AerosolDepositionModelFamily.VENKATRAM_PLEIM_1999:
            if self.surface is not None:
                raise ValueError("VP size integration does not accept a Zhang SurfacePhysicsBundle")
            if self.vp_resistance_provider is None:
                raise ValueError("VP size integration requires an explicit size-resistance provider")
        else:
            raise ValueError("unsupported aerosol deposition model family")

    def _particle(self, mode: LognormalAerosolMode, diameter_m: float) -> AerosolParticleProperties:
        return AerosolParticleProperties(
            diameter_m=diameter_m,
            density_kg_m3=mode.density_kg_m3,
            diameter_basis=mode.diameter_basis,
            provenance=f"{mode.provenance.compact_label} | mode={mode.label}",
        )

    def _vd(self, mode: LognormalAerosolMode, diameter_m: float) -> float:
        particle = self._particle(mode, diameter_m)
        if self.model_family is AerosolDepositionModelFamily.ZHANG2001_SLINN:
            assert self.surface is not None
            return StandaloneZhang2001AerosolDeposition(particle, self.surface, self.meteorology).deposition_velocity_m_s()
        assert self.vp_resistance_provider is not None
        resistance = self.vp_resistance_provider.resistance(particle=particle, meteorology=self.meteorology)
        return StandaloneVenkatramPleimAerosolDeposition(particle, resistance, self.meteorology).deposition_velocity_m_s()

    def _split_log_points(self) -> tuple[float, ...]:
        lo = math.log(self.diameter_domain.min_diameter_m)
        hi = math.log(self.diameter_domain.max_diameter_m)
        points = [lo, hi]
        if self.model_family is AerosolDepositionModelFamily.ZHANG2001_SLINN and self.surface is not None:
            threshold = float(self.surface.aerosol.rebound_activation_diameter_m)
            if self.diameter_domain.min_diameter_m < threshold < self.diameter_domain.max_diameter_m:
                points.append(math.log(threshold))
        return tuple(sorted(set(points)))

    def _mode_average(self, mode: LognormalAerosolMode, weighting: AerosolDistributionWeighting) -> tuple[float, float]:
        retained = mode.retained_fraction(domain=self.diameter_domain, weighting=weighting)
        if retained <= 0.0:
            raise ValueError(f"mode {mode.label!r} has zero {weighting.value}-weighted coverage in diameter domain")
        if mode.geometric_std_dev == 1.0:
            return self._vd(mode, mode.geometric_mean_diameter_m), 1.0

        mu = mode._weighted_mu(weighting)
        s = mode.ln_sigma
        nodes, weights = leggauss(int(self.quadrature_order))
        total = 0.0
        split = self._split_log_points()
        norm = s * math.sqrt(2.0 * math.pi)
        for a, b in zip(split[:-1], split[1:]):
            half = 0.5 * (b - a)
            mid = 0.5 * (a + b)
            xs = mid + half * nodes
            vals = []
            for x in xs:
                xf = float(x)
                d = math.exp(xf)
                pdf_x = math.exp(-0.5 * ((xf - mu) / s) ** 2) / norm
                vals.append(self._vd(mode, d) * pdf_x)
            total += half * float(np.dot(weights, np.asarray(vals, dtype=float)))
        conditional = total / retained
        if not math.isfinite(conditional) or conditional < 0.0:
            raise ArithmeticError("size-integrated deposition velocity is invalid")
        return conditional, retained

    def result(self) -> AerosolSizeDistributionDepositionResult:
        per_mode: list[ModeIntegratedDepositionResult] = []
        for mode in self.distribution.modes:
            vdn, fn = self._mode_average(mode, AerosolDistributionWeighting.NUMBER)
            vdm, fm = self._mode_average(mode, AerosolDistributionWeighting.MASS)
            per_mode.append(ModeIntegratedDepositionResult(
                label=mode.label,
                total_number_concentration_m3=mode.total_number_concentration_m3,
                total_mass_concentration_kg_m3=mode.total_mass_concentration_kg_m3,
                retained_number_fraction=fn,
                retained_mass_fraction=fm,
                in_domain_number_concentration_m3=mode.total_number_concentration_m3*fn,
                in_domain_mass_concentration_kg_m3=mode.total_mass_concentration_kg_m3*fm,
                number_weighted_vd_m_s=vdn,
                mass_weighted_vd_m_s=vdm,
                geometric_mean_diameter_m=mode.geometric_mean_diameter_m,
                mass_geometric_mean_diameter_m=mode.mass_geometric_mean_diameter_m,
                geometric_std_dev=mode.geometric_std_dev,
                density_kg_m3=mode.density_kg_m3,
                diameter_basis=mode.diameter_basis,
                provenance=mode.provenance.compact_label,
            ))

        n_total = self.distribution.total_number_concentration_m3
        m_total = self.distribution.total_mass_concentration_kg_m3
        n_in = math.fsum(r.in_domain_number_concentration_m3 for r in per_mode)
        m_in = math.fsum(r.in_domain_mass_concentration_kg_m3 for r in per_mode)
        if n_in <= 0.0 or m_in <= 0.0:
            raise ValueError("diameter domain contains no represented number or mass")
        vdn = math.fsum(r.in_domain_number_concentration_m3*r.number_weighted_vd_m_s for r in per_mode)/n_in
        vdm = math.fsum(r.in_domain_mass_concentration_kg_m3*r.mass_weighted_vd_m_s for r in per_mode)/m_in
        return AerosolSizeDistributionDepositionResult(
            model_family=self.model_family,
            diameter_basis=self.distribution.diameter_basis,
            domain_min_diameter_m=self.diameter_domain.min_diameter_m,
            domain_max_diameter_m=self.diameter_domain.max_diameter_m,
            retained_number_fraction=n_in/n_total,
            retained_mass_fraction=m_in/m_total,
            number_weighted_vd_m_s=vdn,
            mass_weighted_vd_m_s=vdm,
            total_number_concentration_m3=n_total,
            total_mass_concentration_kg_m3=m_total,
            in_domain_number_concentration_m3=n_in,
            in_domain_mass_concentration_kg_m3=m_in,
            mode_results=tuple(per_mode),
            quadrature_order=int(self.quadrature_order),
            distribution_provenance=self.distribution.provenance.compact_label,
            domain_provenance=self.diameter_domain.provenance.compact_label,
        )
