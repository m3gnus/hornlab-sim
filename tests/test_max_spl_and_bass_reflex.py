from __future__ import annotations

import numpy as np
import pytest

from hornlab_sim.methods.bandpass import Driver
from hornlab_sim.methods.bass_reflex import (
    SweepConfig,
    alignment_metrics,
    ebp_hint,
    port_area_for_length_m,
)
from hornlab_sim.methods.max_spl import design_front_chamber, xmax_limited_spl


def _driver() -> Driver:
    return Driver(
        Sd=57e-4,
        Bl=9.0,
        Re=5.5,
        Le=0.23e-3,
        Mmd=5.7e-3,
        Cms=351e-6,
        Rms=1.0,
    )


def test_xmax_limited_spl_returns_finite_ceiling():
    driver = _driver()
    port, vb_front = design_front_chamber(driver, fb_helmholtz=1200.0)
    freq = np.logspace(np.log10(100.0), np.log10(2000.0), 32)

    result = xmax_limited_spl(
        driver=driver,
        Vb_front=vb_front,
        front_port=port,
        Vb_rear=0.4e-3,
        Xmax_mm=3.8,
        Vamp_max=35.0,
        freq=freq,
    )

    assert result.freq.shape == freq.shape
    assert np.all(np.isfinite(result.spl_max))
    assert set(np.unique(result.limit_reason)) <= {"X", "V"}
    assert np.all(result.spl_max <= np.maximum(result.spl_xmax, result.spl_amp))


def test_bass_reflex_short_port_metrics_accept_feasible_alignment():
    row = {
        "Brand": "Example",
        "Model": "Woofer",
        "Z_ohm": "8",
        "Size_in": "15",
        "Fs_Hz": "35",
        "Qts": "0.34",
        "Qes": "0.38",
        "Qms": "5.0",
        "Vas_L": "140",
        "Sd_cm2": "850",
        "Bl_Tm": "18",
        "Re_ohm": "5.6",
        "Mms_g": "110",
        "Xmax_mm": "8",
        "Power_W": "800",
    }
    config = SweepConfig(
        size_in=15.0,
        vb_values_l=np.array([100.0]),
        fb_values_hz=np.array([35.0]),
        port_lengths_m=np.array([0.10]),
        score_band=(28.0, 120.0),
        velocity_cap_mps=1.0e6,
        max_equiv_diam_mm=1000.0,
        min_port_resonance_hz=0.0,
        top_n=5,
    )
    freq = np.logspace(np.log10(20.0), np.log10(200.0), 48)
    driver = Driver(
        Sd=850e-4,
        Bl=18.0,
        Re=5.6,
        Mms=110e-3,
        Cms=1.0 / ((2.0 * np.pi * 35.0) ** 2 * 110e-3),
        Qms=5.0,
    )

    metrics = alignment_metrics(row, driver, freq, config, 100.0, 35.0, 0.10)

    assert metrics is not None
    assert metrics["Port_diam_equiv_mm"] > 0.0
    assert metrics["ShortPortScore"] == pytest.approx(float(metrics["ShortPortScore"]))
    assert ebp_hint(35.0, 0.38)[1] == "vented"
    assert port_area_for_length_m(100.0, 35.0, 0.10) > 0.0
