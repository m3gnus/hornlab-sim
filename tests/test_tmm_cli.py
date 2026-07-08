from __future__ import annotations

from hornlab_sim.cli.tmm import main
from hornlab_sim.methods.bandpass import Driver


def test_tmm_cli_duct_smoke(capsys):
    rc = main(
        [
            "duct",
            "--width",
            "130",
            "--height",
            "376",
            "--depth",
            "500",
            "--n-points",
            "12",
            "--segments",
            "3",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Rectangular duct" in out
    assert "Quarter-wave prediction" in out


def test_tmm_cli_accepts_project_injected_driver_preset(capsys):
    driver = Driver(
        Sd=530e-4,
        Bl=21.0,
        Re=5.4,
        Le=0.6e-3,
        Mmd=98e-3,
        Cms=180e-6,
        Rms=2.5,
        n_drivers=2,
    )

    rc = main(
        [
            "lf-slot",
            "--openings",
            "130",
            "--n-points",
            "12",
            "--segments",
            "3",
            "--driver",
            "demo",
        ],
        driver_presets={"demo": driver},
        prog="test-tmm",
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TMM peak" in out
    assert "Lumped peak" in out
