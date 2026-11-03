"""Cascaded transfer-matrix model for 1-D acoustic ducts.

Computes input impedance of ducts and cavities by cascading 2×2 plane-wave
transfer matrices through a segmented geometry.  This captures standing-wave
resonances (half-wave cancellations, organ-pipe modes) that a single lumped
compliance element misses.

The module is a drop-in upgrade for any ``Chamber.load_impedance`` call in
:mod:`lumped.bandpass`.  Use ``TMMLoad`` as a duck-typed replacement for
``Chamber`` — it exposes the same ``load_impedance(omega)`` interface.

Physics:  plane-wave propagation with Kirchhoff–Benade viscothermal wall
losses.  Valid up to the first transverse mode of each segment
(f < c / 2W_min).

References:
    Beranek & Mellow, *Acoustics* (2012), ch. 10.
    Keefe, JASA 1984 — viscothermal losses in ducts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from .bandpass import RHO, C_SOUND, Port
from .port_acoustics import (
    CP_AIR as CP_AIR,
    GAMMA_AIR,
    KAPPA_AIR as KAPPA_AIR,
    MU_AIR,
    PRANDTL_AIR,
)

# ── Air thermo-viscous properties (20 °C, 1 atm) ────────────────────────
GAMMA = GAMMA_AIR
PRANDTL = PRANDTL_AIR


# ── Low-level matrix builders ────────────────────────────────────────────

def _complex_wavenumber(
    omega: np.ndarray,
    area: float,
    perimeter: float,
    losses: bool,
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """Plane-wave wavenumber, optionally with Kirchhoff viscothermal losses."""
    k0 = omega / c
    if not losses or perimeter <= 0 or area <= 0:
        return k0

    r_h = 2.0 * area / perimeter  # hydraulic radius
    omega_safe = np.where(omega == 0, 1.0, omega)
    delta_v = np.sqrt(2.0 * MU_AIR / (rho * omega_safe))
    delta_t = delta_v / math.sqrt(PRANDTL)
    alpha = (delta_v / (2.0 * r_h)) * (1.0 + (GAMMA - 1.0) * delta_t / delta_v)
    k = k0 * (1.0 + (1.0 - 1j) * alpha)
    return np.where(omega == 0, 0.0, k)


def uniform_tube_matrix(
    length: float,
    area: float,
    freq: np.ndarray,
    perimeter: float = 0.0,
    losses: bool = True,
    *,
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """2×2 transfer matrix for a uniform tube segment.

    Returns shape ``(len(freq), 2, 2)`` relating ``[p_in, U_in]`` to
    ``[p_out, U_out]`` in the acoustic (pressure / volume-velocity) domain.

    Parameters
    ----------
    length : float
        Segment length (m).
    area : float
        Cross-section area (m²).
    freq : np.ndarray
        Frequency array (Hz).
    perimeter : float
        Wetted perimeter (m) for viscothermal losses.  0 → lossless.
    losses : bool
        Enable viscothermal wall losses.
    rho, c : float
        Air density (kg/m³) and sound speed (m/s) used for propagation and
        characteristic impedance. Other transport constants remain at 20 °C.
    """
    omega = 2.0 * math.pi * freq
    k = _complex_wavenumber(omega, area, perimeter, losses, rho=rho, c=c)
    Z0 = rho * c / area  # characteristic acoustic impedance

    kL = k * length
    cos_kL = np.cos(kL)
    sin_kL = np.sin(kL)

    N = len(freq)
    T = np.empty((N, 2, 2), dtype=complex)
    T[:, 0, 0] = cos_kL
    T[:, 0, 1] = 1j * Z0 * sin_kL
    T[:, 1, 0] = 1j * sin_kL / Z0
    T[:, 1, 1] = cos_kL
    return T


def area_discontinuity_matrix(
    area_in: float,
    area_out: float,
    freq: np.ndarray,
) -> np.ndarray:
    """Return the ideal abrupt-junction matrix in the ``(p, U)`` domain.

    Pressure ``p`` and volume velocity ``U`` are both continuous at an ideal
    area step, so no transformer is inserted between adjacent tube matrices.
    ``area_in`` and ``area_out`` are retained for API compatibility and to
    make the junction geometry explicit. Higher-order evanescent step effects
    are outside this plane-wave model.
    """
    N = len(freq)
    T = np.zeros((N, 2, 2), dtype=complex)
    T[:, 0, 0] = 1.0
    T[:, 1, 1] = 1.0
    return T


# ── Cascade and impedance ───────────────────────────────────────────────

def cascade_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Multiply a sequence of (N_freq, 2, 2) transfer matrices."""
    T = matrices[0].copy()
    for M in matrices[1:]:
        T = np.matmul(T, M)
    return T


def input_impedance(
    T: np.ndarray,
    Z_load: Union[np.ndarray, float, None],
) -> np.ndarray:
    """Input impedance from a cascaded transfer matrix and load impedance.

    Z_in = (T00·Z_L + T01) / (T10·Z_L + T11)

    ``Z_load=None`` → rigid termination: Z_in = T00 / T10.
    """
    if Z_load is None:
        # Rigid wall: Z_load → ∞  →  Z_in = T00 / T10
        return T[:, 0, 0] / T[:, 1, 0]
    return (T[:, 0, 0] * Z_load + T[:, 0, 1]) / (T[:, 1, 0] * Z_load + T[:, 1, 1])


# ── Radiation impedance helpers ──────────────────────────────────────────

def radiation_impedance_baffled(
    area: float,
    freq: np.ndarray,
    *,
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """Baffled-piston radiation impedance (low-ka approximation).

    Uses the equivalent circular radius for the real part (resistance) and
    the Rayleigh end-correction mass for the reactive part.
    """
    omega = 2.0 * math.pi * freq
    a_eq = math.sqrt(area / math.pi)
    R_rad = rho * omega**2 / (2.0 * math.pi * c)
    X_rad = 1j * omega * rho * 0.85 * a_eq / area
    return R_rad + X_rad


def radiation_impedance_unbaffled(
    area: float,
    freq: np.ndarray,
    *,
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """Unbaffled (flanged-free) piston radiation impedance (low-ka approx)."""
    omega = 2.0 * math.pi * freq
    a_eq = math.sqrt(area / math.pi)

    R_rad = rho * omega**2 / (4.0 * math.pi * c)
    X_rad = 1j * omega * rho * 0.61 * a_eq / area
    return R_rad + X_rad


# ── Main user-facing function ────────────────────────────────────────────

def duct_input_impedance(
    areas: Sequence[float],
    perimeters: Sequence[float],
    lengths: Sequence[float],
    freq: np.ndarray,
    *,
    termination: Union[str, np.ndarray] = "rigid",
    termination_area: float = 0.0,
    losses: bool = True,
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """Input impedance of an arbitrary segmented duct.

    The duct is defined by ``N`` segments, each with a cross-section area,
    wetted perimeter, and length.  The segments are cascaded from input
    (index 0) to output (index N-1). No separate area-discontinuity matrix is
    needed: pressure and volume velocity are continuous between adjacent
    segments in this module's ``(p, U)`` state convention.

    Parameters
    ----------
    areas : sequence of float
        Cross-section area of each segment (m²).
    perimeters : sequence of float
        Wetted perimeter of each segment (m²).  For rectangular: 2*(W+H).
    lengths : sequence of float
        Length of each segment (m).
    freq : np.ndarray
        Frequency array (Hz).
    termination : str or np.ndarray
        ``"rigid"`` (closed end), ``"baffled"`` (baffled-piston radiation),
        ``"unbaffled"`` (free-end radiation), or a complex impedance array.
    termination_area : float
        Exit area for radiation termination models (m²).  Ignored for rigid
        or custom array terminations.
    losses : bool
        Enable viscothermal wall losses.
    rho, c : float
        Air density (kg/m³) and sound speed (m/s) for the full duct and any
        built-in radiation termination. Custom termination arrays must already
        use these same air properties.

    Returns
    -------
    np.ndarray
        Complex input impedance, shape ``(len(freq),)``.
    """
    n = len(areas)
    if n != len(perimeters) or n != len(lengths):
        raise ValueError("areas, perimeters, lengths must have equal length")
    if n == 0:
        raise ValueError("at least one segment required")

    T = uniform_tube_matrix(
        lengths[0],
        areas[0],
        freq,
        perimeter=perimeters[0],
        losses=losses,
        rho=rho,
        c=c,
    )
    for i in range(1, n):
        segment = uniform_tube_matrix(
            lengths[i],
            areas[i],
            freq,
            perimeter=perimeters[i],
            losses=losses,
            rho=rho,
            c=c,
        )
        T = np.matmul(T, segment)

    if isinstance(termination, str):
        if termination == "rigid":
            Z_load = None
        elif termination == "baffled":
            Z_load = radiation_impedance_baffled(
                termination_area if termination_area > 0 else areas[-1], freq,
                rho=rho, c=c)
        elif termination == "unbaffled":
            Z_load = radiation_impedance_unbaffled(
                termination_area if termination_area > 0 else areas[-1], freq,
                rho=rho, c=c)
        else:
            raise ValueError(f"unknown termination {termination!r}")
    else:
        Z_load = termination

    return input_impedance(T, Z_load)


# ── Slot-pocket convenience functions ────────────────────────────────────

def slot_pocket_impedance(
    slot_opening_W_mm: float,
    slot_height_mm: float,
    apex_depth_mm: float,
    apex_width_mm: float,
    freq: np.ndarray,
    *,
    n_segments: int = 30,
    losses: bool = True,
    termination: str = "rigid",
    rho: float = RHO,
    c: float = C_SOUND,
) -> np.ndarray:
    """TMM input impedance of one slot pocket.

    The pocket is a tapered rectangular duct: constant height, width
    tapering linearly from ``slot_opening_W`` (mouth) to ``apex_width``
    (back wall).  The back is rigid-terminated by default.

    Returns impedance looking *into* the pocket from the mouth.  The
    caller adds slot-exit radiation impedance and wires into the BP
    network.
    """
    depth_m = apex_depth_mm * 1e-3
    h_m = slot_height_mm * 1e-3
    w_mouth_m = slot_opening_W_mm * 1e-3
    w_back_m = apex_width_mm * 1e-3

    seg_len = depth_m / n_segments
    areas: list[float] = []
    perimeters: list[float] = []
    lengths: list[float] = []

    for i in range(n_segments):
        t = (i + 0.5) / n_segments  # midpoint fraction
        w = w_mouth_m + (w_back_m - w_mouth_m) * t
        a = w * h_m
        p = 2.0 * (w + h_m)
        areas.append(a)
        perimeters.append(p)
        lengths.append(seg_len)

    return duct_input_impedance(
        areas, perimeters, lengths, freq,
        termination=termination, losses=losses, rho=rho, c=c,
    )


def slot_pocket_impedance_from_params(
    params: Any,
    freq: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """TMM slot pocket impedance from a full cabinet params object
    (duck-typed: needs ``params.slot``)."""
    slot = params.slot
    return slot_pocket_impedance(
        slot_opening_W_mm=slot.slot_opening_W,
        slot_height_mm=slot.slot_height,
        apex_depth_mm=slot.resolved_apex_depth(),
        apex_width_mm=slot.apex_width_mm,
        freq=freq,
        **kwargs,
    )


# ── TMMLoad: duck-typed Chamber replacement ──────────────────────────────

@dataclass
class TMMLoad:
    """A chamber-like load whose impedance comes from a precomputed TMM.

    Duck-types ``Chamber.load_impedance(omega, rho=..., c=...)`` so it plugs into
    ``bandpass.simulate()`` without API changes.

    If a ``port`` is provided, the port admittance is added in parallel
    (same topology as ``Chamber.load_impedance``): the cavity and port
    share the same face.

    ``rho`` and ``c`` record the fixed air properties used to build ``_Z_tmm``.
    For arbitrary precomputed arrays the caller must supply that provenance;
    omitted values declare the legacy defaults. A runtime mismatch is rejected:
    rebuild the TMM impedance and load for the requested environment.
    """
    _Z_tmm: np.ndarray
    _omega_ref: np.ndarray
    port: Optional[Port] = None
    rho: float = RHO
    c: float = C_SOUND

    def load_impedance(
        self,
        omega: np.ndarray,
        *,
        rho: float = RHO,
        c: float = C_SOUND,
    ) -> np.ndarray:
        if rho != self.rho or c != self.c:
            raise ValueError(
                f"TMMLoad air properties (rho={rho}, c={c}) differ from "
                f"construction (rho={self.rho}, c={self.c}); rebuild the TMM "
                "impedance and load with the requested rho and c"
            )
        omega_array = np.asarray(omega)
        omega_ref = np.asarray(self._omega_ref)
        if len(omega_array) != len(omega_ref):
            raise ValueError(
                f"TMMLoad was built for {len(omega_ref)} frequency "
                f"points but load_impedance received {len(omega_array)}"
            )
        if omega_array.shape != omega_ref.shape or not np.allclose(
            omega_array,
            omega_ref,
            rtol=1.0e-9,
            atol=0.0,
        ):
            raise ValueError(
                "TMMLoad frequency grid does not match the grid used to "
                "build the load"
            )
        Y = 1.0 / self._Z_tmm
        if self.port is not None:
            Y = Y + 1.0 / self.port.impedance(omega_array, rho=rho, c=c)
        return 1.0 / Y


def make_slot_tmm_load(
    slot_opening_W_mm: float,
    slot_height_mm: float,
    apex_depth_mm: float,
    apex_width_mm: float,
    freq: np.ndarray,
    *,
    exit_port: Optional[Port] = None,
    n_segments: int = 30,
    losses: bool = True,
    rho: float = RHO,
    c: float = C_SOUND,
) -> TMMLoad:
    """Build a fixed-environment slot load; use the same rho/c in simulate()."""
    omega = 2.0 * math.pi * freq
    Z_pocket = slot_pocket_impedance(
        slot_opening_W_mm, slot_height_mm, apex_depth_mm, apex_width_mm,
        freq, n_segments=n_segments, losses=losses, rho=rho, c=c,
    )
    return TMMLoad(_Z_tmm=Z_pocket, _omega_ref=omega, port=exit_port, rho=rho, c=c)


__all__ = [
    "uniform_tube_matrix",
    "area_discontinuity_matrix",
    "cascade_matrices",
    "input_impedance",
    "radiation_impedance_baffled",
    "radiation_impedance_unbaffled",
    "duct_input_impedance",
    "slot_pocket_impedance",
    "slot_pocket_impedance_from_params",
    "TMMLoad",
    "make_slot_tmm_load",
]
