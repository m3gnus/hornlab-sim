"""Validate the lumped BP6S model against a Hornresp export.

Usage:
    python -m hornlab_sim.hornresp.validate CONFIG.txt DATA.txt [--out plot.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from hornlab_sim.methods.bandpass import (
    Driver, Port, Chamber, simulate,
)
from .io import parse_config, parse_response


def build_from_hornresp(cfg) -> tuple[Driver, Chamber, Chamber]:
    """Build the parallel-driver lumped model represented by ``cfg``.

    Multi-driver Hornresp series wiring is rejected because
    :class:`~hornlab_sim.methods.bandpass.Driver` models identical drivers in
    parallel only.
    """
    if cfg.n_drivers > 1 and str(cfg.wiring).upper() == "S":
        raise ValueError(
            "Hornresp series-wired multi-driver configs are unsupported; "
            "the lumped bandpass model supports parallel wiring only"
        )
    driver = Driver(
        Sd=cfg.Sd,
        Bl=cfg.Bl,
        Re=cfg.Re,
        Le=cfg.Le,
        Mmd=cfg.Mmd,
        Cms=cfg.Cms,
        Rms=cfg.Rms,
        n_drivers=cfg.n_drivers,
    )

    # Hornresp horn-segment numbering: Vc1/Ap1/Lp1 = first chamber+port
    # AFTER the driver. For BP6S, Vc2/Ap2/Lp2 mirror that on the rear
    # side. For BP4 (sealed rear), Hornresp leaves Ap2=Lp2=0 and Vc2 is
    # the sealed rear volume — handle that as Chamber(port=None).
    front_port = Port(area=cfg.Ap1, length=max(cfg.Lp1, 0.0)) \
                 if cfg.Ap1 > 0 else None
    rear_port = Port(area=cfg.Ap2, length=max(cfg.Lp2, 0.0)) \
                if cfg.Ap2 > 0 else None

    front_chamber = Chamber(volume=cfg.Vc1, port=front_port)
    rear_chamber = Chamber(volume=cfg.Vc2, port=rear_port)
    return driver, front_chamber, rear_chamber


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("data")
    p.add_argument("--out", default=None,
                   help="optional PNG output path for the comparison plot")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    cfg = parse_config(args.config)
    f_hr, spl_hr, phase_hr = parse_response(args.data)

    print(f"Loaded Hornresp config:")
    topo = ("BP6S" if cfg.BP6S else "BP4" if cfg.BP4 else "—")
    print(f"  Topology          : {topo}, n_drivers={cfg.n_drivers}, "
          f"wiring='{cfg.wiring or '—'}'")
    print(f"  Sd                : {cfg.Sd*1e4:.1f} cm²")
    print(f"  Bl, Re, Le        : {cfg.Bl} T·m, {cfg.Re} Ω, {cfg.Le*1e3:.2f} mH")
    print(f"  Mmd, Cms, Rms     : {cfg.Mmd*1e3:.2f} g, {cfg.Cms:.3e} m/N, {cfg.Rms:.2f} N·s/m")
    print(f"  Front  Vc1/Ap1/Lp1: {cfg.Vc1*1e3:.2f} L / {cfg.Ap1*1e4:.1f} cm² / {cfg.Lp1*1e2:.2f} cm")
    print(f"  Rear   Vc2/Ap2/Lp2: {cfg.Vc2*1e3:.2f} L / {cfg.Ap2*1e4:.1f} cm² / {cfg.Lp2*1e2:.2f} cm")
    print(f"  Eg, Rg, Ω         : {cfg.Eg} V, {cfg.Rg} Ω, {cfg.Ang_steradians/np.pi:.2f}π")

    driver, front_ch, rear_ch = build_from_hornresp(cfg)

    sim = simulate(
        driver=driver,
        front_chamber=front_ch,
        rear_chamber=rear_ch,
        freq=f_hr,
        Rg=cfg.Rg,
        v_g=cfg.Eg,
        radiation_half_space=cfg.half_space,
        distance_m=1.0,
    )

    # Derived diagnostic quantities
    if front_ch.port is not None:
        fb_front = 1.0 / (2 * np.pi * np.sqrt(front_ch.port.Mport * front_ch.Cab))
        print(f"\nFront port × front chamber Helmholtz: {fb_front:6.1f} Hz "
              f"(Mport={front_ch.port.Mport:.2f} kg/m^4)")
    else:
        fb_front = float("inf")
        print("\nFront chamber sealed.")
    if rear_ch.port is not None:
        fb_rear = 1.0 / (2 * np.pi * np.sqrt(rear_ch.port.Mport * rear_ch.Cab))
        print(f"Rear  port × rear  chamber Helmholtz: {fb_rear:6.1f} Hz "
              f"(Mport={rear_ch.port.Mport:.2f} kg/m^4)")
    else:
        print("Rear chamber sealed.")

    # Auto-pick comparison band: span the bulk of the data, but cap at
    # the front-chamber upper Helmholtz × 1.5 (above which 1D chamber
    # modes start to dominate Hornresp's response).
    f_lim = min(fb_front * 2.0, f_hr.max() * 0.9)
    mask = (f_hr >= 20.0) & (f_hr <= f_lim)
    err = sim.spl_total[mask] - spl_hr[mask]
    rms = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    print(f"\nFit (20 Hz – {f_lim:.0f} Hz):")
    print(f"  RMS error : {rms:.2f} dB")
    print(f"  Mean bias : {bias:+.2f} dB")
    print(f"  Max |err| : {np.max(np.abs(err)):.2f} dB at "
          f"{f_hr[mask][np.argmax(np.abs(err))]:.1f} Hz")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
        return 0

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.semilogx(f_hr, spl_hr, label="Hornresp", lw=1.5)
    ax.semilogx(sim.freq, sim.spl_total, label="lumped BP6S",
                lw=1.5, ls="--")
    if sim.spl_rear_port is not None:
        ax.semilogx(sim.freq, sim.spl_front_port, label="front port only",
                    lw=0.8, alpha=0.6)
        ax.semilogx(sim.freq, sim.spl_rear_port, label="rear port only",
                    lw=0.8, alpha=0.6)
    ax.set_ylabel("SPL (dB) @ 1 m / 2.83 V")
    ax.set_xlim(10, 2000)
    ax.set_ylim(60, 110)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", ncol=2)
    ax.set_title("BP6S frequency response — lumped vs Hornresp")
    if np.isfinite(fb_front):
        ax.axvline(fb_front, color="gray", ls=":", alpha=0.5)
    if rear_ch.port is not None:
        ax.axvline(fb_rear, color="gray", ls=":", alpha=0.5)

    ax = axes[1]
    ax.semilogx(sim.freq, np.abs(sim.Z_electrical), label="|Z| lumped")
    ax.set_ylabel("Electrical |Z| (Ω)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(10, 2000)

    ax = axes[2]
    ax.semilogx(sim.freq, sim.cone_excursion_mm, label="cone excursion @ Eg")
    ax.set_ylabel("Cone excursion (mm peak)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(10, 2000)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=140)
        print(f"\nSaved plot → {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
