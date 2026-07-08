from __future__ import annotations

import numpy as np
import pytest

from hornlab_sim.hornresp import (
    build_from_hornresp,
    export_bp4,
    parse_config,
    parse_response,
)
from hornlab_sim.methods.bandpass import Driver, Port


def test_hornresp_bp4_export_parse_round_trip(tmp_path):
    driver = Driver(
        Sd=57e-4,
        Bl=9.0,
        Re=5.5,
        Le=0.23e-3,
        Mmd=5.7e-3,
        Cms=351e-6,
        Rms=1.0,
    )
    out = tmp_path / "bp4.txt"

    export_bp4(
        driver,
        Vb_front=36.32e-6,
        front_port=Port(area=57e-4 / 8.0, length=15e-3),
        Vb_rear=0.4e-3,
        path=out,
        comment="round trip",
    )

    raw = out.read_bytes()
    assert b"\r\n" in raw
    cfg = parse_config(out)
    parsed_driver, front_chamber, rear_chamber = build_from_hornresp(cfg)

    assert cfg.BP4 is True
    assert cfg.BP6S is False
    assert cfg.Sd == pytest.approx(driver.Sd)
    assert parsed_driver.Sd == pytest.approx(driver.Sd)
    assert front_chamber.volume == pytest.approx(36.32e-6, abs=0.03e-6)
    assert front_chamber.port.area == pytest.approx(57e-4 / 8.0)
    assert rear_chamber.port is None


def test_hornresp_response_parser_reads_three_columns(tmp_path):
    response = tmp_path / "response.txt"
    response.write_text(
        "Freq\tSPL\tWPhase\n"
        "20\t80.0\t0\n"
        "40\t86.0\t-15\n",
        encoding="utf-8",
    )

    freq, spl, phase = parse_response(response)

    assert np.allclose(freq, [20.0, 40.0])
    assert np.allclose(spl, [80.0, 86.0])
    assert np.allclose(phase, [0.0, -15.0])
