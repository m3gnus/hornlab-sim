from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_sim.methods.bandpass import Driver
from hornlab_sim.methods.bass_reflex import (
    SweepConfig,
    alignment_metrics,
    ebp_hint,
    fval,
    port_area_for_length_m,
    row_to_driver,
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


def _woofer_row() -> dict[str, str]:
    return {
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
    np.testing.assert_array_equal(
        result.spl_max,
        np.minimum(result.spl_xmax, result.spl_amp),
    )


@pytest.mark.parametrize(
    ("sd", "fb", "port_ratio", "length", "expected_hex"),
    [
        (
            57e-4,
            1200.0,
            1.0 / 8.0,
            15e-3,
            (
                "0x1.758e219652bd4p-11",
                "0x1.eb851eb851eb8p-7",
                "0x1.30a5d721f4d60p-15",
            ),
        ),
        (
            850e-4,
            45.0,
            0.2,
            0.1,
            (
                "0x1.16872b020c49cp-6",
                "0x1.999999999999ap-4",
                "0x1.c7542d04dea04p-4",
            ),
        ),
        (
            0.0123456789,
            987.654321,
            0.137,
            0.023456789,
            (
                "0x1.bb611d57d4657p-10",
                "0x1.8050e767f2716p-6",
                "0x1.588c932e12354p-14",
            ),
        ),
    ],
)
def test_design_front_chamber_remains_bit_identical(
    sd,
    fb,
    port_ratio,
    length,
    expected_hex,
):
    driver = Driver(
        Sd=sd,
        Bl=1.0,
        Re=1.0,
        Mms=1.0,
        Cms=1.0,
        Rms=1.0,
    )

    port, volume = design_front_chamber(
        driver,
        fb_helmholtz=fb,
        port_ratio=port_ratio,
        Lp_phys=length,
    )

    assert (port.area.hex(), port.length.hex(), volume.hex()) == expected_hex


def test_bass_reflex_short_port_metrics_accept_feasible_alignment():
    row = _woofer_row()
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


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_bass_reflex_csv_parser_treats_nonfinite_values_as_missing(value):
    row = _woofer_row()
    row["Fs_Hz"] = value

    assert fval(row, "Fs_Hz") == 0.0
    assert fval(row, "Fs_Hz", 12.5) == 12.5
    assert row_to_driver(row) is None
    assert ebp_hint(float(value), 0.38) == (None, "unknown")


@pytest.mark.parametrize(
    ("vb_l", "fb_hz", "lp_m"),
    [
        (math.inf, 35.0, 0.1),
        (100.0, math.inf, 0.1),
        (100.0, 35.0, math.inf),
        (math.nan, 35.0, 0.1),
    ],
)
def test_port_area_rejects_nonfinite_inputs(vb_l, fb_hz, lp_m):
    assert port_area_for_length_m(vb_l, fb_hz, lp_m) is None
