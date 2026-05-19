"""Canonical port acoustics primitives.

Single source of truth for the Rayleigh end-correction constants and the
inverse port-length-for-Fb formula used by every lumped/parametric module
in MEH-Lab. Importers:

    - ``lumped.bandpass`` (``Port.end_correction``)
    - ``bigmeh_parametric.helmholtz`` (``end_correction``, slot/mid helpers)
    - ``bigmeh_design.derive`` (``port_length_for_fb`` and split variant)

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Rayleigh end-correction factor for a flanged (baffled) circular piston.
#: ΔL_flanged = FLANGED_END_CORRECTION_FACTOR * r, with r = sqrt(A/π).
FLANGED_END_CORRECTION_FACTOR: float = 0.85

#: Rayleigh end-correction factor for an unflanged (free / open) circular
#: piston. ΔL_free = FREE_END_CORRECTION_FACTOR * r.
FREE_END_CORRECTION_FACTOR: float = 0.61


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
        ``"baffled"``        — alias for ``"flanged"``; lets BIGMEH slot-exit
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


# ---------------------------------------------------------------------------
# Inverse port tuning
# ---------------------------------------------------------------------------

def port_length_for_fb(
    area_m2: float,
    Vb_m3: float,
    fb_hz: float,
    c_sound: float = 343.0,
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
    "end_correction",
    "port_length_for_fb",
]
