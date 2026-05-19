"""Helmholtz resonance helpers.

Provides:
    helmholtz(V, A, L_geom=0, end_corr="flanged_free", c=343) -> Hz
    bigmeh_slot_helmholtz(opening_W_mm, slot_depth_mm, slot_height_mm,
                          apex_width_mm=0, apex_depth_mm=slot_depth_mm,
                          interpretation="slot_pocket", ...) -> dict

Reference (omnicalculator):
    f = (c / 2π) · √( A / (V · L_eff) ),  L_eff = L_geom + ΔL

Where ΔL is the end-correction term. Standard Rayleigh values:
    flanged (baffled) end:    0.85 · r
    unflanged (free) end:     0.61 · r
    r = √(A/π)  (equivalent radius of the port opening)

A baffled hole (port with no neck) opening into open air on one side and a
cavity on the other gets ΔL ≈ 0.85·r + 0.61·r = 1.46·r.

For BIGMEH slot-loaded cabinets, the lumped Helmholtz model is a stretch
because the slot is a tapered wedge, not a constant-section neck. The most
useful interpretation is "slot_pocket": the wedge pocket is the cavity (V),
the slot exit is a baffled opening (A), and L_eff = end correction only.
This matches the ~250 Hz tune observed for a 130 mm × 376 mm × 500 mm-deep
slot at 12.22 L pocket volume.
"""

from __future__ import annotations

import math
from typing import Dict, Literal

from .port_acoustics import end_correction  # noqa: F401  (re-export)


C_AIR = 343.0  # m/s


def helmholtz(
    V_m3: float,
    A_m2: float,
    L_geom_m: float = 0.0,
    end_corr: str = "flanged_free",
    c: float = C_AIR,
    n_parallel: int = 1,
) -> float:
    """Helmholtz resonator frequency in Hz.

    f = (c / 2π) · √( A_total / (V · L_eff) ),  L_eff = L_geom + ΔL(end_corr)

    For n parallel identical ports (each of area A_total/n) sharing one
    cavity, the equivalent acoustic mass is `M_eq = ρ · L_eff_per_port /
    A_total`, with `L_eff_per_port = L_geom + ΔL(per-port radius)`. This
    leaves the formula above unchanged — only `ΔL` shrinks because it is
    computed from the per-port equivalent radius `sqrt((A/n)/π)`. For a
    split-into-n single combined opening the modeled frequency is therefore
    higher than the naive combined-area answer by the ΔL ratio.

    Args:
        V_m3:       cavity volume (m³)
        A_m2:       total port cross-section area (m²)
        L_geom_m:   geometric port length (m); 0 for a hole in a baffle
        end_corr:   end-correction mode; see end_correction() docstring
        c:          speed of sound (m/s); default 343
        n_parallel: count of equal-area parallel openings sharing the
                    cavity. Defaults to 1 (single port). Use n=2 for the
                    BIGMEH split slot pair / Option C front-baffle ports.
    """
    if V_m3 <= 0 or A_m2 <= 0:
        raise ValueError(f"V and A must be positive: V={V_m3}, A={A_m2}")
    L_eff = L_geom_m + end_correction(A_m2, end_corr, n_parallel=n_parallel)
    if L_eff <= 0:
        raise ValueError("L_eff must be positive (set end_corr or L_geom_m > 0)")
    return (c / (2.0 * math.pi)) * math.sqrt(A_m2 / (V_m3 * L_eff))


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


def bigmeh_slot_helmholtz(
    opening_W_mm: float,
    slot_depth_mm: float = 500.0,
    slot_height_mm: float = 376.0,
    apex_width_mm: float = 0.0,
    apex_depth_mm: float | None = None,
    interpretation: Literal[
        "slot_pocket", "back_cavity_long_port", "back_cavity_thin_port"
    ] = "slot_pocket",
    end_corr: str = "flanged_free",
    cabinet_W_mm: float = 800.0,
    cabinet_H_mm: float = 400.0,
    cabinet_D_mm: float = 600.0,
    c: float = C_AIR,
    slot_topology: str = "front",
) -> Dict[str, float]:
    """Compute a Helmholtz approximation for one BIGMEH slot.

    The lumped Helmholtz model is topology-independent: the pocket volume
    and exit area are the same for front-firing and side-firing slots
    (identical cross-section, just rotated). ``slot_topology`` is accepted
    for API completeness and echoed in the result dict.

    The BIGMEH cabinet is bandpass-loaded (driver on inclined slot wall has
    a sealed back chamber + slot-port front chamber), so a single Helmholtz
    figure is always an approximation. Pick the interpretation that matches
    the question:

    "slot_pocket" (default, matches user's expected ~250 Hz for 130 mm):
        V = slot pocket volume from its trapezoid cross-section
        (½·(opening_W + apex_width)·apex_depth·slot_height);
        A = slot exit area; L_geom = 0; the slot exit acts as a baffled hole
        with end correction. The default apex_width=0 and apex_depth=slot_depth
        preserves the original triangular pocket model. Each slot pocket has
        one driver behind it (n_parallel=1); the slot exit is a single hole.

    "back_cavity_long_port" (textbook ported-box reading):
        V = cabinet − 2·slot pockets (the chamber behind the drivers);
        A = both slot exit areas combined; L_geom = slot_depth. Two slots
        share the rear chamber, so n_parallel=2 — the end correction is
        computed from each slot's own area, not the combined area. Treats
        the whole slot as a long port — typically too low.

    "back_cavity_thin_port" (curiosity):
        Same V as above, A = both slot exits, L_geom = 0; relies on end
        correction only. n_parallel=2 for the same reason as above. Sits
        in between the others.
    """
    opening_W_mm = _require_finite_positive("opening_W_mm", opening_W_mm)
    slot_depth_mm = _require_finite_positive("slot_depth_mm", slot_depth_mm)
    slot_height_mm = _require_finite_positive("slot_height_mm", slot_height_mm)
    apex_depth_mm = (slot_depth_mm if apex_depth_mm is None
                     else _require_finite_positive("apex_depth_mm", apex_depth_mm))
    apex_width_mm = _require_finite_nonnegative("apex_width_mm", apex_width_mm)
    if apex_width_mm > opening_W_mm:
        raise ValueError(
            f"apex_width_mm {apex_width_mm} must be <= opening_W_mm {opening_W_mm}"
        )
    A_one = (opening_W_mm * slot_height_mm) * 1e-6  # m²
    A_both = 2.0 * A_one
    A_xs_mm2 = 0.5 * (opening_W_mm + apex_width_mm) * apex_depth_mm
    V_pocket_one = A_xs_mm2 * slot_height_mm * 1e-9
    V_cab = cabinet_W_mm * cabinet_H_mm * cabinet_D_mm * 1e-9
    V_back = V_cab - 2.0 * V_pocket_one

    if interpretation == "slot_pocket":
        V = V_pocket_one
        A = A_one
        L_geom = 0.0
        n_parallel = 1
    elif interpretation == "back_cavity_long_port":
        V = V_back
        A = A_both
        L_geom = slot_depth_mm * 1e-3
        n_parallel = 2
    elif interpretation == "back_cavity_thin_port":
        V = V_back
        A = A_both
        L_geom = 0.0
        n_parallel = 2
    else:
        raise ValueError(f"unknown interpretation {interpretation!r}")

    delta = end_correction(A, end_corr, n_parallel=n_parallel)
    L_eff = L_geom + delta
    f = helmholtz(V, A, L_geom, end_corr, c, n_parallel=n_parallel)
    return {
        "interpretation": interpretation,
        "end_corr_mode": end_corr,
        "n_parallel": n_parallel,
        "V_m3": V,
        "V_L": V * 1000.0,
        "A_m2": A,
        "L_geom_m": L_geom,
        "L_eff_m": L_eff,
        "delta_L_m": delta,
        "f_Hz": f,
        # Echoed inputs for context
        "opening_W_mm": opening_W_mm,
        "slot_depth_mm": slot_depth_mm,
        "slot_height_mm": slot_height_mm,
        "apex_width_mm": apex_width_mm,
        "apex_depth_mm": apex_depth_mm,
        "slot_cross_section_mm2": A_xs_mm2,
        "V_pocket_one_L": V_pocket_one * 1000.0,
        "V_back_L": V_back * 1000.0,
        "slot_topology": slot_topology,
    }


def bigmeh_slot_helmholtz_from_params(
    params: "BigMEHParams",
    **kwargs,
) -> Dict[str, float]:
    """Compute the BIGMEH slot Helmholtz approximation from BigMEHParams."""
    return bigmeh_slot_helmholtz(
        opening_W_mm=params.slot.slot_opening_W,
        slot_depth_mm=params.slot.slot_depth,
        slot_height_mm=params.slot.slot_height,
        apex_width_mm=params.slot.apex_width_mm,
        apex_depth_mm=params.slot.resolved_apex_depth(),
        cabinet_W_mm=params.cabinet_W,
        cabinet_H_mm=params.cabinet_H,
        cabinet_D_mm=params.cabinet_D,
        slot_topology=params.slot_topology,
        **kwargs,
    )


def bigmeh_mid_chamber_helmholtz(
    mids: "MidSectionParams | None" = None,
    *,
    chamber_volume_cc: float | None = None,
    port_count: int | None = None,
    entry_area_cm2: float | None = None,
    chamber_area_cm2: float | None = None,
    tube_depth_mm: float | None = None,
    target_fc_hz: float | None = None,
    end_corr: str = "flanged_free",
    c: float = C_AIR,
) -> Dict[str, object]:
    """Compute the closed-form Helmholtz estimate for one mid front chamber.

    This is a first-pass sizing helper, not a final tuning oracle. BIGMEH
    research-doc §2a / §2d cites CAFMEH #75 and Ingard 1953: the Rayleigh
    closed form over-shoots the chamber-volume-to-frequency shift by ~1.75×,
    so the BEM-resolved f_H usually sits roughly 10-15% lower than this
    estimate. The mid-port geometry also follows Danley US6411718 col. 8-10:
    use conical/frustum tap channels with a conical or near-conical horn body.
    """
    if mids is None:
        from .params import MidSectionParams

        mids = MidSectionParams()

    mids.driver.validate()
    mids.port.validate()
    mids.chamber.validate(mids.driver, mids.port, mids.target_fc_hz)

    port_count = (
        mids.port.count_per_chamber if port_count is None else int(port_count)
    )
    if port_count < 1:
        raise ValueError(f"port_count must be >= 1, got {port_count!r}")
    entry_area_cm2 = (
        mids.port.entry_area_cm2
        if entry_area_cm2 is None
        else _require_finite_positive("entry_area_cm2", entry_area_cm2)
    )
    chamber_area_cm2 = (
        mids.port.chamber_area_cm2
        if chamber_area_cm2 is None
        else _require_finite_positive("chamber_area_cm2", chamber_area_cm2)
    )
    tube_depth_mm = (
        mids.port.tube_depth_mm
        if tube_depth_mm is None
        else _require_finite_nonnegative("tube_depth_mm", tube_depth_mm)
    )
    target_fc_hz = mids.target_fc_hz if target_fc_hz is None else target_fc_hz
    target_fc_hz = _require_finite_positive("target_fc_hz", target_fc_hz)
    chamber_volume_cc = (
        mids.chamber.resolved_volume_cc(
            mids.driver,
            mids.port,
            mids.target_fc_hz,
        )
        if chamber_volume_cc is None
        else _require_finite_positive("chamber_volume_cc", chamber_volume_cc)
    )

    A_one = entry_area_cm2 * 1e-4
    A = A_one * port_count
    V = chamber_volume_cc * 1e-6
    L_geom = tube_depth_mm * 1e-3
    delta = end_correction(A, end_corr, n_parallel=port_count)
    L_eff = L_geom + delta
    f = helmholtz(V, A, L_geom, end_corr, c, n_parallel=port_count)

    cylinder_radius = None
    cylinder_depth = None
    if mids.chamber.shape == "cylinder":
        cylinder_radius = mids.chamber.resolved_cylinder_radius_mm(mids.driver)
        cylinder_depth = mids.chamber.resolved_cylinder_depth_mm(
            mids.driver,
            mids.port,
            mids.target_fc_hz,
        )

    return {
        "end_corr_mode": end_corr,
        "n_parallel": port_count,
        "V_m3": V,
        "V_L": V * 1000.0,
        "V_cc": chamber_volume_cc,
        "A_m2": A,
        "A_one_m2": A_one,
        "L_geom_m": L_geom,
        "L_eff_m": L_eff,
        "delta_L_m": delta,
        "f_Hz": f,
        "target_fc_hz": target_fc_hz,
        "entry_area_cm2": entry_area_cm2,
        "entry_total_area_cm2": entry_area_cm2 * port_count,
        "chamber_area_cm2": chamber_area_cm2,
        "flare_ratio": chamber_area_cm2 / entry_area_cm2,
        "tube_depth_mm": tube_depth_mm,
        "chamber_shape": mids.chamber.shape,
        "cylinder_radius_mm": cylinder_radius,
        "cylinder_depth_mm": cylinder_depth,
        "formula_bias_note": (
            "research-doc §2a/§2d: CAFMEH #75 / Ingard 1953 indicate "
            "the closed form over-shoots the V→f_H shift by ~1.75x; "
            "BEM/measurement should set final tuning"
        ),
    }


def bigmeh_mid_chamber_helmholtz_from_params(
    params: "BigMEHParams",
    **kwargs,
) -> Dict[str, object]:
    """Compute one BIGMEH mid-chamber Helmholtz estimate from BigMEHParams."""
    return bigmeh_mid_chamber_helmholtz(params.mids, **kwargs)


def main() -> None:
    """CLI: print a comparison table for slot widths."""
    import argparse
    p = argparse.ArgumentParser(prog="bigmeh_parametric.helmholtz")
    p.add_argument("--mid", action="store_true",
                   help="Print the default mid-chamber estimate instead")
    p.add_argument("--openings", type=str, default="80,130,200",
                   help="Comma-separated slot opening widths in mm")
    p.add_argument("--slot-depth", type=float, default=500.0)
    p.add_argument("--slot-height", type=float, default=376.0)
    p.add_argument("--apex-width", type=float, default=0.0)
    p.add_argument("--apex-depth", type=float, default=None)
    p.add_argument("--cabinet-W", type=float, default=800.0)
    p.add_argument("--cabinet-H", type=float, default=400.0)
    p.add_argument("--cabinet-D", type=float, default=600.0)
    p.add_argument("--end-corr", type=str, default="flanged_free")
    args = p.parse_args()

    if args.mid:
        r = bigmeh_mid_chamber_helmholtz(end_corr=args.end_corr)
        print(f"{'target':>8} | {'V (cc)':>8} | {'A_one (cm²)':>11} | "
              f"{'ports':>5} | {'L_geom (mm)':>11} | {'L_eff (mm)':>10} | "
              f"{'f (Hz)':>7}")
        print("-" * 78)
        print(f"{r['target_fc_hz']:>8.0f} | {r['V_cc']:>8.1f} | "
              f"{r['entry_area_cm2']:>11.2f} | {r['n_parallel']:>5} | "
              f"{r['L_geom_m']*1000:>11.1f} | "
              f"{r['L_eff_m']*1000:>10.1f} | {r['f_Hz']:>7.1f}")
        return

    openings = [float(x) for x in args.openings.split(",")]
    interps = ("slot_pocket", "back_cavity_long_port", "back_cavity_thin_port")

    print(f"{'opening':>9} | {'interpretation':>23} | "
          f"{'V (L)':>8} | {'A (m²)':>8} | "
          f"{'L_geom (mm)':>11} | {'L_eff (mm)':>10} | {'f (Hz)':>7}")
    print("-" * 95)
    for w in openings:
        for interp in interps:
            r = bigmeh_slot_helmholtz(
                opening_W_mm=w,
                slot_depth_mm=args.slot_depth,
                slot_height_mm=args.slot_height,
                apex_width_mm=args.apex_width,
                apex_depth_mm=args.apex_depth,
                cabinet_W_mm=args.cabinet_W,
                cabinet_H_mm=args.cabinet_H,
                cabinet_D_mm=args.cabinet_D,
                interpretation=interp,
                end_corr=args.end_corr,
            )
            print(f"{w:>9.0f} | {interp:>23} | "
                  f"{r['V_L']:>8.2f} | {r['A_m2']:>8.4f} | "
                  f"{r['L_geom_m']*1000:>11.0f} | {r['L_eff_m']*1000:>10.0f} | "
                  f"{r['f_Hz']:>7.1f}")
        print()


if __name__ == "__main__":
    main()
