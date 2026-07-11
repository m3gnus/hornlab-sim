"""Canonical port acoustics primitives.

Single source of truth for the Rayleigh end-correction constants and the
inverse port-length-for-Fb formula used by every lumped/parametric module
in this stack. Importers:

    - ``hornlab_sim.methods.bandpass`` (``Port.end_correction``)
    - ``hornlab_sim.methods.helmholtz`` (``end_correction``, slot/mid helpers)
    - upstream cabinet-design derivation tooling (``port_length_for_fb``
      and split variant)

Rationale (Rayleigh):
    For a circular piston of radius r the radiated reactive air-mass adds
    an effective length ΔL beyond the geometric port length. The standard
    textbook values are

        flanged (baffled) end:    ΔL ≈ 0.85 · r
        unflanged (free)  end:    ΔL ≈ 0.61 · r
        r = sqrt(A / π)           (equivalent radius for area A)

    A hole in an infinite baffle opening into a cavity on the other side
    therefore picks up roughly 0.85·r + 0.61·r = 1.46·r in total end
    correction (one flanged, one free).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Rayleigh end-correction factor for a flanged (baffled) circular piston.
#: ΔL_flanged = FLANGED_END_CORRECTION_FACTOR * r, with r = sqrt(A/π).
FLANGED_END_CORRECTION_FACTOR: float = 0.85

#: Rayleigh end-correction factor for an unflanged (free / open) circular
#: piston. ΔL_free = FREE_END_CORRECTION_FACTOR * r.
FREE_END_CORRECTION_FACTOR: float = 0.61

#: Sound speed used by the legacy lumped port-tuning formulae.
SPEED_OF_SOUND: float = 343.0

# Air thermo-viscous properties at roughly 20 C, 1 atm. These mirror the
# Kirchhoff-Benade constants used in transfer_matrix.py.
MU_AIR: float = 1.846e-5
KAPPA_AIR: float = 0.0257
CP_AIR: float = 1005.0
GAMMA_AIR: float = 1.4
PRANDTL_AIR: float = MU_AIR * CP_AIR / KAPPA_AIR


def _require_finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value


def _require_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return value


def _end_factor(mode: str) -> float:
    if mode in ("flanged", "baffled"):
        return FLANGED_END_CORRECTION_FACTOR
    if mode == "free":
        return FREE_END_CORRECTION_FACTOR
    raise ValueError(f"unknown local end-correction mode {mode!r}")


# ---------------------------------------------------------------------------
# End correction
# ---------------------------------------------------------------------------

def end_correction(
    area_m2: float,
    mode: str = "flanged_free",
    n_parallel: int = 1,
) -> float:
    """Return ΔL (m) for an opening of cross-section ``area_m2``.

    ``n_parallel`` handles split-but-identical ports: ΔL is computed from
    the per-port equivalent radius ``sqrt((A/n)/π)``, not the combined
    radius. For ``n_parallel=1`` the result matches the single-port
    Rayleigh formulae below.

    Modes:
        ``"none"``           — no correction (ΔL = 0)
        ``"flanged"``        — single flanged end, 0.85·r (one face baffled,
                               other into cavity)
        ``"free"``           — single unflanged end, 0.61·r
        ``"baffled"``        — alias for ``"flanged"``; lets slot-exit
                               callers spell their physical intent.
        ``"flanged_both"``   — flanged at both ends, 1.7·r (tube between two
                               baffled openings or two cavities)
        ``"flanged_flanged"`` — alias for ``"flanged_both"`` (used in mid-
                               port / Helmholtz-tube callers)
        ``"free_both"``      — free at both ends, 1.22·r
        ``"flanged_free"``   — one flanged, one free, 1.46·r (the BAFFLED
                               HOLE in a wall — most ports default to this)
    """
    if n_parallel < 1:
        raise ValueError(f"n_parallel must be >= 1, got {n_parallel!r}")
    r = math.sqrt((area_m2 / n_parallel) / math.pi)
    if mode == "none":
        return 0.0
    if mode in ("flanged", "baffled"):
        return FLANGED_END_CORRECTION_FACTOR * r
    if mode == "free":
        return FREE_END_CORRECTION_FACTOR * r
    if mode in ("flanged_both", "flanged_flanged"):
        return 2.0 * FLANGED_END_CORRECTION_FACTOR * r
    if mode == "free_both":
        return 2.0 * FREE_END_CORRECTION_FACTOR * r
    if mode == "flanged_free":
        return (FLANGED_END_CORRECTION_FACTOR + FREE_END_CORRECTION_FACTOR) * r
    raise ValueError(f"unknown end-correction mode {mode!r}")


def end_correction_for_radius(radius_m: float, mode: str) -> float:
    """Return a local Rayleigh end correction for one circular end."""
    radius_m = _require_finite_positive("radius_m", radius_m)
    if mode == "none":
        return 0.0
    return _end_factor(mode) * radius_m


def end_correction_terms(
    entry_radius_m: float,
    exit_radius_m: float,
    mode: str = "flanged_free",
) -> tuple[float, float]:
    """Return ``(entry_delta_m, exit_delta_m)`` for non-uniform ports.

    ``entry`` is the exterior/horn-side end and ``exit`` is the chamber-side
    end. The combined modes mirror :func:`end_correction` while applying each
    term to its local radius.
    """
    entry_radius_m = _require_finite_positive("entry_radius_m", entry_radius_m)
    exit_radius_m = _require_finite_positive("exit_radius_m", exit_radius_m)

    if mode == "none":
        return 0.0, 0.0
    if mode in ("flanged", "baffled"):
        return _end_factor("flanged") * entry_radius_m, 0.0
    if mode == "free":
        return _end_factor("free") * entry_radius_m, 0.0
    if mode in ("flanged_both", "flanged_flanged"):
        return (
            _end_factor("flanged") * entry_radius_m,
            _end_factor("flanged") * exit_radius_m,
        )
    if mode == "free_both":
        return (
            _end_factor("free") * entry_radius_m,
            _end_factor("free") * exit_radius_m,
        )
    if mode == "flanged_free":
        return (
            _end_factor("free") * entry_radius_m,
            _end_factor("flanged") * exit_radius_m,
        )
    raise ValueError(f"unknown end-correction mode {mode!r}")


def confined_interior_end_correction(
    radius_m: float,
    chamber_volume_m3: float,
    *,
    base_mode: str = "flanged",
    confinement_factor: float = 9.0 / 8.0,
    max_confinement: float = 0.85,
) -> float:
    """Ingard-style chamber-side confined-neck end correction.

    Ingard, "On the Theory and Design of Acoustic Resonators", JASA 25
    (1953), treats the neck reactance as geometry-dependent rather than a
    universal Rayleigh constant. For a neck opening into a finite chamber,
    the chamber-side velocity field is increasingly confined as the aperture
    radius ``a`` approaches the chamber length scale
    ``R_v = (3V / 4π)^(1/3)``. This helper applies that confined-neck scaling
    to the local Rayleigh term with a bounded Padé form:

        ΔL_i = k_i a / (1 - β a / R_v)

    where ``k_i`` is the local Rayleigh factor, ``β = 9/8`` is the
    first-order confined-neck scale used by the accompanying research notes, and
    the denominator is clamped so tiny chambers cannot produce a singular
    lumped correction. Large chambers approach Rayleigh-like behavior as
    ``a/R_v → 0``; the confinement scaling itself never switches off, so
    this is a permanent geometry-dependent model, not an asymptotic patch
    on the Rayleigh constant.
    """
    radius_m = _require_finite_positive("radius_m", radius_m)
    chamber_volume_m3 = _require_finite_positive(
        "chamber_volume_m3", chamber_volume_m3,
    )
    confinement_factor = _require_finite_positive(
        "confinement_factor", confinement_factor,
    )
    max_confinement = _require_finite_positive("max_confinement", max_confinement)
    if max_confinement >= 1.0:
        raise ValueError("max_confinement must be < 1")

    chamber_radius_m = (3.0 * chamber_volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    x = min(max_confinement, confinement_factor * radius_m / chamber_radius_m)
    return _end_factor(base_mode) * radius_m / (1.0 - x)


def frustum_port_acoustic_mass(
    a_entry_m: float,
    a_exit_m: float,
    length_m: float,
    rho: float,
) -> float:
    """Acoustic mass of a linear-radius frustum port.

    For ``a(x)`` varying linearly from ``a_entry`` to ``a_exit``,
    ``M = rho * integral(dx / A(x)) = rho*L/(pi*a_entry*a_exit)``.
    """
    a_entry_m = _require_finite_positive("a_entry_m", a_entry_m)
    a_exit_m = _require_finite_positive("a_exit_m", a_exit_m)
    length_m = _require_finite_nonnegative("length_m", length_m)
    rho = _require_finite_positive("rho", rho)
    return rho * length_m / (math.pi * a_entry_m * a_exit_m)


def frustum_port_inertance_denominator(
    a_entry_m: float,
    a_exit_m: float,
    length_m: float,
    *,
    end_corr: str = "flanged_free",
    interior_end_correction: str = "rayleigh",
    chamber_volume_m3: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(integral dx/A, entry_delta, exit_delta)`` for a frustum.

    The first value has units ``1/m`` and is the quantity used by the
    Helmholtz frequency formula:

        f = c/(2π) * sqrt(1 / (V * [integral dx/A + Σ ΔL_i/A_i]))
    """
    a_entry_m = _require_finite_positive("a_entry_m", a_entry_m)
    a_exit_m = _require_finite_positive("a_exit_m", a_exit_m)
    length_m = _require_finite_nonnegative("length_m", length_m)
    entry_delta_m, exit_delta_m = end_correction_terms(
        a_entry_m,
        a_exit_m,
        end_corr,
    )
    if interior_end_correction == "ingard" and exit_delta_m > 0.0:
        if chamber_volume_m3 is None:
            raise ValueError(
                "chamber_volume_m3 is required for interior_end_correction='ingard'"
            )
        # Preserve the local end type while applying the confined-cavity scale.
        base_mode = "flanged" if exit_delta_m >= FLANGED_END_CORRECTION_FACTOR * a_exit_m * 0.999 else "free"
        exit_delta_m = confined_interior_end_correction(
            min(a_entry_m, a_exit_m),
            chamber_volume_m3,
            base_mode=base_mode,
        )
    elif interior_end_correction != "rayleigh":
        raise ValueError(
            "interior_end_correction must be 'rayleigh' or 'ingard', "
            f"got {interior_end_correction!r}"
        )

    A_entry = math.pi * a_entry_m * a_entry_m
    A_exit = math.pi * a_exit_m * a_exit_m
    denom = length_m / (math.pi * a_entry_m * a_exit_m)
    denom += entry_delta_m / A_entry
    denom += exit_delta_m / A_exit
    return denom, entry_delta_m, exit_delta_m


def port_hydraulic_radius(
    area_m2: float,
    *,
    perimeter_m: float | None = None,
    n_parallel: int = 1,
) -> float:
    """Hydraulic radius scale used by the Kirchhoff-Benade loss model.

    The transfer-matrix module uses ``r_h = 2A/P`` for a duct segment. If no
    perimeter is supplied, this falls back to a circular port with the
    per-port area ``area_m2 / n_parallel``.
    """
    area_m2 = _require_finite_positive("area_m2", area_m2)
    if n_parallel < 1:
        raise ValueError(f"n_parallel must be >= 1, got {n_parallel!r}")
    area_one = area_m2 / n_parallel
    if perimeter_m is not None:
        perimeter_one = _require_finite_positive("perimeter_m", perimeter_m)
        return 2.0 * area_one / perimeter_one
    return math.sqrt(area_one / math.pi)


def viscothermal_port_q(
    frequency_hz: float | np.ndarray,
    area_m2: float,
    *,
    perimeter_m: float | None = None,
    hydraulic_radius_m: float | None = None,
    n_parallel: int = 1,
    rho: float = 1.21,
    mu: float = MU_AIR,
    gamma: float = GAMMA_AIR,
    prandtl: float = PRANDTL_AIR,
) -> float | np.ndarray:
    """Kirchhoff-Benade effective viscothermal Q for a port tube.

    The same boundary-layer terms as ``transfer_matrix._complex_wavenumber``
    are converted to a series-loss quality factor:

        δ_v = sqrt(2μ/(ρω))
        δ_t = δ_v/sqrt(Pr)
        Q ≈ r_h / [δ_v * (1 + (γ - 1)/sqrt(Pr))]

    ``r_h`` follows the transfer-matrix convention ``2A/P``. For circular
    ports with no supplied perimeter this reduces to the equivalent radius.
    """
    rho = _require_finite_positive("rho", rho)
    mu = _require_finite_positive("mu", mu)
    gamma = _require_finite_positive("gamma", gamma)
    prandtl = _require_finite_positive("prandtl", prandtl)
    if hydraulic_radius_m is None:
        r_h = port_hydraulic_radius(
            area_m2,
            perimeter_m=perimeter_m,
            n_parallel=n_parallel,
        )
    else:
        r_h = _require_finite_positive("hydraulic_radius_m", hydraulic_radius_m)

    freq = np.asarray(frequency_hz, dtype=float)
    if np.any(freq <= 0) or np.any(~np.isfinite(freq)):
        raise ValueError(f"frequency_hz must be positive and finite, got {frequency_hz!r}")
    omega = 2.0 * math.pi * freq
    delta_v = np.sqrt(2.0 * mu / (rho * omega))
    q = r_h / (delta_v * (1.0 + (gamma - 1.0) / math.sqrt(prandtl)))
    if np.ndim(frequency_hz) == 0:
        return float(q)
    return q


# ---------------------------------------------------------------------------
# Inverse port tuning
# ---------------------------------------------------------------------------

def port_length_for_fb(
    area_m2: float,
    Vb_m3: float,
    fb_hz: float,
    c_sound: float = SPEED_OF_SOUND,
    flanged_each_end: bool = True,
) -> tuple[float, float]:
    """Solve the Helmholtz formula for the physical port length.

    Returns ``(L_physical_m, L_effective_m)``. The effective length includes
    the end correction at both ends of the port (``flanged_each_end``
    selects 0.85·r or 0.61·r per end). Negative ``L_physical_m`` means the
    geometry has no room for the requested Fb — caller has to widen the
    chamber or shrink the port area.
    """
    omega_b = 2.0 * math.pi * fb_hz
    L_eff = (c_sound ** 2 * area_m2) / (omega_b ** 2 * Vb_m3)
    a = math.sqrt(area_m2 / math.pi)
    factor = (
        FLANGED_END_CORRECTION_FACTOR
        if flanged_each_end
        else FREE_END_CORRECTION_FACTOR
    )
    end_corr_each = factor * a
    return L_eff - 2.0 * end_corr_each, L_eff


__all__ = [
    "FLANGED_END_CORRECTION_FACTOR",
    "FREE_END_CORRECTION_FACTOR",
    "SPEED_OF_SOUND",
    "MU_AIR",
    "KAPPA_AIR",
    "CP_AIR",
    "GAMMA_AIR",
    "PRANDTL_AIR",
    "end_correction",
    "end_correction_for_radius",
    "end_correction_terms",
    "confined_interior_end_correction",
    "frustum_port_acoustic_mass",
    "frustum_port_inertance_denominator",
    "port_hydraulic_radius",
    "viscothermal_port_q",
    "port_length_for_fb",
]
