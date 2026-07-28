"""Xmax- and voltage-limited maximum SPL helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bandpass import Driver, Port, bp4_sealed_rear, log_freq
from .port_acoustics import SPEED_OF_SOUND, end_correction


@dataclass
class MaxSplResult:
    freq: np.ndarray
    spl_at_drive: np.ndarray
    spl_xmax: np.ndarray
    spl_amp: np.ndarray
    spl_max: np.ndarray
    drive_v_xmax: np.ndarray
    cone_excursion_mm_at_drive: np.ndarray
    Xmax_mm: float
    Vamp_max: float
    limit_reason: np.ndarray


def design_front_chamber(
    driver: Driver,
    fb_helmholtz: float = 1200.0,
    port_ratio: float = 1.0 / 8.0,
    Lp_phys: float = 15e-3,
) -> tuple[Port, float]:
    """Size a BP4 front port/chamber from a Helmholtz target."""
    Sp = driver.Sd * port_ratio
    L_eff = Lp_phys + end_correction(Sp, mode="flanged_both")
    Vb = (SPEED_OF_SOUND ** 2 * Sp) / (
        4 * math.pi ** 2 * L_eff * fb_helmholtz ** 2
    )
    return Port(area=Sp, length=Lp_phys), Vb


def xmax_limited_spl(
    driver: Driver,
    Vb_front: float,
    front_port: Port,
    Vb_rear: float,
    Xmax_mm: float,
    Vamp_max: float = 35.0,
    *,
    freq: np.ndarray | None = None,
    v_g_ref: float = 2.83,
) -> MaxSplResult:
    """Scale a BP4 response to the cone-excursion and voltage ceiling."""
    if freq is None:
        freq = log_freq(50, 5000, 1500)
    sim = bp4_sealed_rear(
        driver=driver,
        Vb_front=Vb_front,
        front_port=front_port,
        Vb_rear=Vb_rear,
        freq=freq,
        v_g=v_g_ref,
    )
    drive_scale_xmax = Xmax_mm / np.maximum(sim.cone_excursion_mm, 1e-12)
    spl_xmax = sim.spl_total + 20 * np.log10(drive_scale_xmax)

    drive_scale_amp = Vamp_max / v_g_ref
    spl_amp = sim.spl_total + 20 * np.log10(drive_scale_amp)

    limit_reason = np.where(drive_scale_xmax <= drive_scale_amp, "X", "V")

    return MaxSplResult(
        freq=freq,
        spl_at_drive=sim.spl_total,
        spl_xmax=spl_xmax,
        spl_amp=spl_amp,
        spl_max=np.minimum(spl_xmax, spl_amp),
        drive_v_xmax=v_g_ref * drive_scale_xmax,
        cone_excursion_mm_at_drive=sim.cone_excursion_mm,
        Xmax_mm=Xmax_mm,
        Vamp_max=Vamp_max,
        limit_reason=limit_reason,
    )


def alpha_alignments(Vas: float) -> list[tuple[str, float]]:
    """Sealed-rear starter volumes parameterized by alpha = Vas / Vb."""
    return [
        ("alpha=4  very stiff", Vas / 4),
        ("alpha=2  moderate", Vas / 2),
        ("alpha=1  standard", Vas),
        ("alpha=0.5 large rear", Vas * 2),
    ]


__all__ = [
    "MaxSplResult",
    "alpha_alignments",
    "design_front_chamber",
    "xmax_limited_spl",
]
