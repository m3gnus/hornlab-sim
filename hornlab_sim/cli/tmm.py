#!/usr/bin/env python3
"""CLI for transfer-matrix acoustic impedance / frequency response.

Templates
---------
lf-slot     Tapered slot pocket → BP4 with TMM front load.
mid-chamber Cylindrical/box front chamber → BP4 with TMM front load.
duct        Generic rectangular duct with arbitrary termination.

Each template computes impedance, optionally runs a full BP4 simulation
(when driver TS params are supplied), and prints a summary table with
standing-wave mode frequencies.

Examples
--------
    # Slot pocket impedance for three opening widths
    python -m hornlab_sim.cli.tmm lf-slot --openings 80,130,200

    # Generic duct impedance
    python -m hornlab_sim.cli.tmm duct --width 130 --height 376 --depth 500 --termination rigid

    # Export impedance data as .npz
    python -m hornlab_sim.cli.tmm lf-slot --openings 130 --out /tmp/slot_tmm.npz
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence

import numpy as np

from hornlab_sim.methods.bandpass import (
    Chamber, Driver, Port, RHO, C_SOUND, log_freq, simulate,
)
from hornlab_sim.methods.transfer_matrix import (
    TMMLoad, duct_input_impedance, make_slot_tmm_load, slot_pocket_impedance,
)


# ── Impedance analysis helpers ───────────────────────────────────────────

def find_impedance_minima(freq: np.ndarray, Z: np.ndarray,
                          threshold_ratio: float = 0.3) -> list[dict]:
    """Find local minima in |Z| that represent standing-wave modes."""
    mag = np.abs(Z)
    minima = []
    for i in range(1, len(mag) - 1):
        if mag[i] < mag[i - 1] and mag[i] < mag[i + 1]:
            if mag[i] < threshold_ratio * np.median(mag):
                minima.append({
                    "freq_hz": float(freq[i]),
                    "Z_mag": float(mag[i]),
                    "Z_phase_deg": float(np.degrees(np.angle(Z[i]))),
                })
    return minima


def print_impedance_summary(label: str, freq: np.ndarray, Z: np.ndarray,
                            depth_mm: float):
    """Print a compact impedance summary with mode frequencies."""
    minima = find_impedance_minima(freq, Z)
    qw_pred = C_SOUND / (4.0 * depth_mm * 1e-3)

    print(f"\n  {label}")
    print(f"    Depth: {depth_mm:.0f} mm")
    print(f"    Quarter-wave prediction: {qw_pred:.0f} Hz")
    if minima:
        for j, m in enumerate(minima):
            order = 2 * j + 1  # odd harmonics for rigid-terminated tube
            print(f"    Mode {order}/4λ: {m['freq_hz']:.0f} Hz  "
                  f"|Z| = {m['Z_mag']:.0f}")
    else:
        print("    No standing-wave modes found in frequency range")


# ── Template: lf-slot ────────────────────────────────────────────────────

def run_lf_slot(args):
    openings = [float(x) for x in args.openings.split(",")]
    freq = log_freq(args.f_min, args.f_max, args.n_points)
    omega = 2.0 * math.pi * freq

    driver_presets = getattr(args, "driver_presets", {})
    if args.driver and args.driver not in driver_presets:
        raise ValueError(f"unknown driver preset {args.driver!r}")
    driver = driver_presets.get(args.driver) if args.driver else None

    print(f"{'opening':>9} | {'depth':>7} | {'V (L)':>7} | "
          f"{'λ/4 (Hz)':>8} | {'1st dip (Hz)':>12} | {'|Z| at dip':>10}")
    print("-" * 72)

    all_results = {}
    for w in openings:
        depth = args.slot_depth
        h = args.slot_height
        apex_w = args.apex_width

        Z = slot_pocket_impedance(
            w, h, depth, apex_w, freq,
            n_segments=args.segments, losses=not args.lossless,
        )

        V_m3 = 0.5 * (w + apex_w) * depth * h * 1e-9
        qw = C_SOUND / (4.0 * depth * 1e-3)
        minima = find_impedance_minima(freq, Z)
        dip_f = minima[0]["freq_hz"] if minima else float("nan")
        dip_Z = minima[0]["Z_mag"] if minima else float("nan")

        print(f"{w:>9.0f} | {depth:>7.0f} | {V_m3*1e3:>7.2f} | "
              f"{qw:>8.0f} | {dip_f:>12.0f} | {dip_Z:>10.0f}")

        key = f"slot_{w:.0f}mm"
        all_results[f"{key}_Z"] = Z
        all_results[f"{key}_freq"] = freq

        if driver is not None:
            A_slot_m2 = (w * 1e-3) * (h * 1e-3)
            exit_port = Port(area=A_slot_m2, length=0.0,
                             flanged_inside=True, flanged_outside=True)
            tmm_load = make_slot_tmm_load(
                w, h, depth, apex_w, freq,
                exit_port=exit_port, n_segments=args.segments,
                losses=not args.lossless,
            )

            Vb_rear_m3 = (args.cabinet_W * args.cabinet_D * args.cabinet_H
                          * 1e-9 - 2.0 * V_m3)
            Vb_rear_m3 = max(Vb_rear_m3, 0.01)

            sim_tmm = simulate(
                driver=driver.derive(),
                front_chamber=tmm_load,
                rear_chamber=Chamber(volume=Vb_rear_m3, port=None),
                freq=freq,
            )

            sim_lump = simulate(
                driver=driver.derive(),
                front_chamber=Chamber(volume=V_m3,
                                      port=Port(area=A_slot_m2, length=0.0,
                                                flanged_inside=True,
                                                flanged_outside=True)),
                rear_chamber=Chamber(volume=Vb_rear_m3, port=None),
                freq=freq,
            )

            all_results[f"{key}_spl_tmm"] = sim_tmm.spl_total
            all_results[f"{key}_spl_lumped"] = sim_lump.spl_total
            all_results[f"{key}_spl_freq"] = freq

            peak_tmm = sim_tmm.spl_total.max()
            peak_f_tmm = freq[np.argmax(sim_tmm.spl_total)]
            peak_lump = sim_lump.spl_total.max()
            peak_f_lump = freq[np.argmax(sim_lump.spl_total)]
            print(f"          | TMM peak: {peak_tmm:.1f} dB @ {peak_f_tmm:.0f} Hz  "
                  f"| Lumped peak: {peak_lump:.1f} dB @ {peak_f_lump:.0f} Hz")

    if args.out:
        np.savez(args.out, **all_results)
        print(f"\nSaved → {args.out}")

    if args.plot or args.save_plot:
        _plot_lf_slot(openings, all_results, freq, driver is not None, args)


def _plot_lf_slot(openings, results, freq, has_spl, args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plot)")
        return

    n_rows = 2 if has_spl else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax_z = axes[0]
    for w in openings:
        key = f"slot_{w:.0f}mm"
        Z = results[f"{key}_Z"]
        ax_z.semilogy(freq, np.abs(Z), label=f"{w:.0f} mm", lw=1.4)
    ax_z.set_ylabel("|Z| (Pa·s/m³)")
    ax_z.set_title("Slot pocket input impedance (TMM)")
    ax_z.legend()
    ax_z.grid(True, which="both", alpha=0.3)

    if has_spl:
        ax_s = axes[1]
        for w in openings:
            key = f"slot_{w:.0f}mm"
            ax_s.semilogx(freq, results[f"{key}_spl_tmm"],
                          label=f"{w:.0f} mm TMM", lw=1.4)
            ax_s.semilogx(freq, results[f"{key}_spl_lumped"],
                          label=f"{w:.0f} mm lumped", lw=0.8, ls="--",
                          alpha=0.6)
        ax_s.set_ylabel("SPL (dB) @ 1 m / 2.83 V")
        ax_s.set_title("BP4 frequency response — TMM vs lumped")
        ax_s.set_ylim(60, 120)
        ax_s.legend(loc="lower center", ncol=3)
        ax_s.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=140)
        print(f"Plot saved → {args.save_plot}")
    if args.plot:
        plt.show()


# ── Template: mid-chamber ────────────────────────────────────────────────

def run_mid_chamber(args):
    freq = log_freq(args.f_min, args.f_max, args.n_points)

    depth_m = args.depth * 1e-3
    if args.shape == "cylinder":
        radius_m = args.radius * 1e-3
        area = math.pi * radius_m ** 2
        perimeter = 2.0 * math.pi * radius_m
    else:
        w_m = args.width * 1e-3
        h_m = args.height * 1e-3
        area = w_m * h_m
        perimeter = 2.0 * (w_m + h_m)

    n_seg = args.segments
    seg_len = depth_m / n_seg
    areas = [area] * n_seg
    perimeters = [perimeter] * n_seg
    lengths = [seg_len] * n_seg

    Z = duct_input_impedance(
        areas, perimeters, lengths, freq,
        termination="rigid", losses=not args.lossless,
    )

    print(f"Mid front chamber ({args.shape})")
    if args.shape == "cylinder":
        print(f"  Radius: {args.radius:.0f} mm, Depth: {args.depth:.0f} mm")
        print(f"  Volume: {area * depth_m * 1e6:.1f} cc")
    else:
        print(f"  {args.width:.0f} × {args.height:.0f} × {args.depth:.0f} mm")
        print(f"  Volume: {area * depth_m * 1e6:.1f} cc")

    print_impedance_summary("Chamber cavity", freq, Z, args.depth)

    if args.out:
        np.savez(args.out, freq=freq, Z=Z)
        print(f"\nSaved → {args.out}")

    if args.plot or args.save_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not available)")
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(freq, np.abs(Z), lw=1.4)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Z| (Pa·s/m³)")
        ax.set_title(f"Mid chamber impedance ({args.shape}, "
                     f"depth={args.depth:.0f} mm)")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=140)
            print(f"Plot saved → {args.save_plot}")
        if args.plot:
            plt.show()


# ── Template: duct ───────────────────────────────────────────────────────

def run_duct(args):
    freq = log_freq(args.f_min, args.f_max, args.n_points)

    w_m = args.width * 1e-3
    h_m = args.height * 1e-3
    depth_m = args.depth * 1e-3
    area = w_m * h_m
    perimeter = 2.0 * (w_m + h_m)

    n_seg = args.segments
    seg_len = depth_m / n_seg
    areas = [area] * n_seg
    perimeters = [perimeter] * n_seg
    lengths = [seg_len] * n_seg

    Z = duct_input_impedance(
        areas, perimeters, lengths, freq,
        termination=args.termination, losses=not args.lossless,
    )

    print(f"Rectangular duct: {args.width:.0f} × {args.height:.0f} × "
          f"{args.depth:.0f} mm, termination={args.termination}")
    print_impedance_summary("Duct", freq, Z, args.depth)

    if args.out:
        np.savez(args.out, freq=freq, Z=Z)
        print(f"\nSaved → {args.out}")

    if args.plot or args.save_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not available)")
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(freq, np.abs(Z), lw=1.4)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Z| (Pa·s/m³)")
        ax.set_title(f"Duct impedance ({args.width:.0f}×{args.height:.0f}×"
                     f"{args.depth:.0f} mm, {args.termination})")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=140)
            print(f"Plot saved → {args.save_plot}")
        if args.plot:
            plt.show()


# ── Argument parser ──────────────────────────────────────────────────────

def build_parser(
    driver_presets: Mapping[str, Driver] | None = None,
    *,
    prog: str = "hornlab-tmm",
) -> argparse.ArgumentParser:
    driver_presets = dict(driver_presets or {})
    p = argparse.ArgumentParser(
        prog=prog,
        description="Transfer-matrix acoustic impedance / frequency response.",
    )
    p.set_defaults(driver_presets=driver_presets)
    sub = p.add_subparsers(dest="template", required=True)

    # ── lf-slot ──
    lf = sub.add_parser("lf-slot", help="Tapered slot pocket BP4")
    lf.add_argument("--openings", type=str, default="80,130,200",
                    help="Comma-separated slot opening widths (mm)")
    lf.add_argument("--slot-depth", type=float, default=500.0, dest="slot_depth")
    lf.add_argument("--slot-height", type=float, default=376.0, dest="slot_height")
    lf.add_argument("--apex-width", type=float, default=0.0, dest="apex_width")
    driver_choices = sorted(driver_presets) or None
    lf.add_argument("--driver", type=str, default=None,
                    choices=driver_choices,
                    help="Driver preset for full BP4 simulation")
    lf.add_argument("--cabinet-W", type=float, default=800.0, dest="cabinet_W")
    lf.add_argument("--cabinet-D", type=float, default=600.0, dest="cabinet_D")
    lf.add_argument("--cabinet-H", type=float, default=400.0, dest="cabinet_H")
    _add_common_args(lf)
    lf.set_defaults(func=run_lf_slot)

    # ── mid-chamber ──
    mc = sub.add_parser("mid-chamber", help="Mid front chamber")
    mc.add_argument("--shape", choices=["cylinder", "box"], default="cylinder")
    mc.add_argument("--radius", type=float, default=50.0,
                    help="Cylinder radius (mm), used when shape=cylinder")
    mc.add_argument("--width", type=float, default=80.0,
                    help="Box width (mm), used when shape=box")
    mc.add_argument("--height", type=float, default=80.0,
                    help="Box height (mm), used when shape=box")
    mc.add_argument("--depth", type=float, default=60.0,
                    help="Chamber depth (mm)")
    _add_common_args(mc)
    mc.set_defaults(func=run_mid_chamber)

    # ── duct ──
    d = sub.add_parser("duct", help="Generic rectangular duct")
    d.add_argument("--width", type=float, required=True, help="Width (mm)")
    d.add_argument("--height", type=float, required=True, help="Height (mm)")
    d.add_argument("--depth", type=float, required=True, help="Depth (mm)")
    d.add_argument("--termination", type=str, default="rigid",
                   choices=["rigid", "baffled", "unbaffled"])
    _add_common_args(d)
    d.set_defaults(func=run_duct)

    return p


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--f-min", type=float, default=20.0, dest="f_min")
    parser.add_argument("--f-max", type=float, default=1000.0, dest="f_max")
    parser.add_argument("--n-points", type=int, default=2000, dest="n_points")
    parser.add_argument("--segments", type=int, default=30)
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz file path")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plot")
    parser.add_argument("--save-plot", type=str, default=None, dest="save_plot",
                        help="Save plot to file")


def main(
    argv: Sequence[str] | None = None,
    *,
    driver_presets: Mapping[str, Driver] | None = None,
    prog: str = "hornlab-tmm",
) -> int:
    p = build_parser(driver_presets, prog=prog)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m hornlab_sim.cli.tmm"))
