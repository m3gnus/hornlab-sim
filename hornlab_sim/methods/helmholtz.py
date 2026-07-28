"""Helmholtz resonance helpers.

Provides:
    helmholtz(V, A, L_geom=0, end_corr="flanged_free", c=343) -> Hz
    slot_helmholtz(opening_W_mm, slot_depth_mm, slot_height_mm,
                   apex_width_mm=0, apex_depth_mm=slot_depth_mm,
                   interpretation="slot_pocket", ...) -> dict
    mid_chamber_helmholtz(...) -> dict

Reference (omnicalculator):
    f = (c / 2π) · √( A / (V · L_eff) ),  L_eff = L_geom + ΔL

Where ΔL is the end-correction term. Standard Rayleigh values:
    flanged (baffled) end:    0.85 · r
    unflanged (free) end:     0.61 · r
    r = √(A/π)  (equivalent radius of the port opening)

A baffled hole (port with no neck) opening into open air on one side and a
cavity on the other gets ΔL ≈ 0.85·r + 0.61·r = 1.46·r.

For slot-loaded cabinets, the lumped Helmholtz model is a stretch
because the slot is a tapered wedge, not a constant-section neck. The most
useful interpretation is "slot_pocket": the wedge pocket is the cavity (V),
the slot exit is a baffled opening (A), and L_eff = end correction only.
This matches the ~250 Hz tune observed for a 130 mm × 376 mm × 500 mm-deep
slot at 12.22 L pocket volume.

The ``*_from_params`` variants take a duck-typed full-cabinet parameter
object (annotated ``CabinetParams`` below): any object exposing ``.slot``,
``.mids``, ``.cabinet_W/H/D`` and ``.slot_topology`` the way an upstream
parametric cabinet model does.

Provenance aliases: these helpers were extracted from a project-specific
package and were first published as ``bigmeh_slot_helmholtz``,
``bigmeh_slot_helmholtz_from_params``, ``bigmeh_mid_chamber_helmholtz``
and ``bigmeh_mid_chamber_helmholtz_from_params``. Those names remain
importable as thin deprecated aliases that emit ``DeprecationWarning``.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Literal

from .port_acoustics import (
    confined_interior_end_correction,
    end_correction,  # noqa: F401  (re-export)
    end_correction_terms,
    frustum_port_acoustic_mass,  # noqa: F401  (re-export)
    frustum_port_inertance_denominator,
    viscothermal_port_q,
)


C_AIR = 343.0  # m/s
RHO_AIR = 1.21  # kg/m^3


def helmholtz(
    V_m3: float,
    A_m2: float,
    L_geom_m: float = 0.0,
    end_corr: str = "flanged_free",
    c: float = C_AIR,
    n_parallel: int = 1,
    rho: float = RHO_AIR,
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
                    cavity. Defaults to 1 (single port). Use n=2 for a
                    split slot pair / paired front-baffle ports.
        rho:        air density (kg/m³), accepted for API symmetry with
                    mass/compliance callers. The density cancels out of the
                    closed-form frequency when c is supplied.
    """
    del rho
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


def _helmholtz_from_inertance_denominator(
    V_m3: float,
    denom_m_inv: float,
    c: float,
) -> float:
    if V_m3 <= 0 or denom_m_inv <= 0:
        raise ValueError("V_m3 and inertance denominator must be positive")
    return (c / (2.0 * math.pi)) * math.sqrt(1.0 / (V_m3 * denom_m_inv))


def _uniform_port_terms(
    A_total_m2: float,
    L_geom_m: float,
    end_corr: str,
    n_parallel: int,
    *,
    chamber_volume_m3: float,
    interior_end_correction: str,
) -> tuple[float, float, float, float]:
    A_one = A_total_m2 / n_parallel
    radius = math.sqrt(A_one / math.pi)
    entry_delta, exit_delta = end_correction_terms(radius, radius, end_corr)
    if interior_end_correction == "ingard" and exit_delta > 0.0:
        base_mode = "flanged" if exit_delta >= 0.85 * radius * 0.999 else "free"
        exit_delta = confined_interior_end_correction(
            radius,
            chamber_volume_m3,
            base_mode=base_mode,
        )
    elif interior_end_correction != "rayleigh":
        raise ValueError(
            "interior_end_correction must be 'rayleigh' or 'ingard', "
            f"got {interior_end_correction!r}"
        )
    L_eff = L_geom_m + entry_delta + exit_delta
    if L_eff <= 0:
        raise ValueError("L_eff must be positive (set end_corr or L_geom_m > 0)")
    return L_eff / A_total_m2, entry_delta, exit_delta, L_eff


def slot_helmholtz(
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
    rho: float = RHO_AIR,
    slot_topology: str = "front",
) -> Dict[str, float]:
    """Compute a Helmholtz approximation for one slot-pocket (front cavity
    + baffled hole).

    The lumped Helmholtz model is topology-independent: the pocket volume
    and exit area are the same for front-firing and side-firing slots
    (identical cross-section, just rotated). ``slot_topology`` is accepted
    for API completeness and echoed in the result dict.

    A slot-loaded cabinet is bandpass-loaded (driver on inclined slot wall
    has a sealed back chamber + slot-port front chamber), so a single
    Helmholtz figure is always an approximation. Pick the interpretation
    that matches the question:

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
    f = helmholtz(V, A, L_geom, end_corr, c, n_parallel=n_parallel, rho=rho)
    A_one = A / n_parallel
    perimeter_one = 2.0 * (
        opening_W_mm * 1e-3 + slot_height_mm * 1e-3
    )
    hydraulic_radius = 2.0 * A_one / perimeter_one
    q_port = viscothermal_port_q(
        f,
        A,
        hydraulic_radius_m=hydraulic_radius,
        n_parallel=n_parallel,
        rho=rho,
    )
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
        "Q_port_derived": q_port,
        "Q_port_eval_hz": f,
        "Q_port_hydraulic_radius_m": hydraulic_radius,
        "Q_port_loss_model": "kirchhoff_benade",
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


def slot_helmholtz_from_params(
    params: "CabinetParams",
    **kwargs,
) -> Dict[str, float]:
    """Compute the slot-pocket Helmholtz approximation from a full cabinet
    params object (duck-typed; see module docstring)."""
    return slot_helmholtz(
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


def mid_chamber_helmholtz(
    mids: "MidSectionParams | None" = None,
    *,
    chamber_volume_cc: float | None = None,
    port_count: int | None = None,
    entry_area_cm2: float | None = None,
    chamber_area_cm2: float | None = None,
    tube_depth_mm: float | None = None,
    entry_radius_m: float | None = None,
    exit_radius_m: float | None = None,
    target_fc_hz: float | None = None,
    end_corr: str = "flanged_free",
    c: float = C_AIR,
    rho: float = RHO_AIR,
    port_model: Literal["uniform", "frustum"] = "uniform",
    interior_end_correction: Literal["rayleigh", "ingard"] = "rayleigh",
    loss_eval_frequency_hz: float | None = None,
    loss_hydraulic_radius_m: float | None = None,
) -> Dict[str, object]:
    """Compute the closed-form Helmholtz estimate for one mid front chamber.

    This is a first-pass sizing helper, not a final tuning oracle. The
    accompanying research notes (§2a / §2d) cite measurements from an
    archived build thread and Ingard 1953: the Rayleigh closed form
    over-shoots the chamber-volume-to-frequency shift by ~1.75×, so the
    BEM-resolved f_H usually sits roughly 10-15% lower than this estimate.
    The mid-port geometry also follows Danley US6411718 col. 8-10:
    use conical/frustum tap channels with a conical or near-conical horn body.

    ``interior_end_correction="ingard"`` is opt-in. It replaces the
    chamber-side Rayleigh term with the confined-neck correction described by
    Ingard, "On the Theory and Design of Acoustic Resonators", JASA 25
    (1953): the interior correction scales with the aperture radius relative
    to the chamber volume length scale, so it grows as small chambers confine
    the neck velocity field. This compresses the chamber-volume-to-frequency
    sensitivity seen in the measured clay-volume regression from an archived
    build thread.

    ``port_model="frustum"`` evaluates the port mass by
    ``integral dx/A(x)`` for a linear-radius taper:
    ``M = rho*L/(pi*a_entry*a_exit)``. End corrections use the local entry
    and chamber-side radii.
    """
    explicit_mid = mids is None
    if mids is not None:
        mids.driver.validate()
        mids.port.validate()
        mids.chamber.validate(mids.driver, mids.port, mids.target_fc_hz)

    if mids is None:
        port_count = 2 if port_count is None else port_count
        entry_area_cm2 = 6.0 if entry_area_cm2 is None else entry_area_cm2
        chamber_area_cm2 = 14.0 if chamber_area_cm2 is None else chamber_area_cm2
        tube_depth_mm = 24.0 if tube_depth_mm is None else tube_depth_mm
        target_fc_hz = 1200.0 if target_fc_hz is None else target_fc_hz

    port_count = int(
        mids.port.count_per_chamber
        if port_count is None and mids is not None
        else port_count
    )
    if port_count < 1:
        raise ValueError(f"port_count must be >= 1, got {port_count!r}")
    entry_area_cm2 = (
        mids.port.entry_area_cm2
        if entry_area_cm2 is None and mids is not None
        else _require_finite_positive("entry_area_cm2", entry_area_cm2)
    )
    chamber_area_cm2 = (
        mids.port.chamber_area_cm2
        if chamber_area_cm2 is None and mids is not None
        else _require_finite_positive("chamber_area_cm2", chamber_area_cm2)
    )
    tube_depth_mm = (
        mids.port.tube_depth_mm
        if tube_depth_mm is None and mids is not None
        else _require_finite_nonnegative("tube_depth_mm", tube_depth_mm)
    )
    if entry_radius_m is not None:
        entry_radius_m = _require_finite_positive("entry_radius_m", entry_radius_m)
        entry_area_cm2 = math.pi * entry_radius_m * entry_radius_m * 1e4
    if exit_radius_m is not None:
        exit_radius_m = _require_finite_positive("exit_radius_m", exit_radius_m)
        chamber_area_cm2 = math.pi * exit_radius_m * exit_radius_m * 1e4
    target_fc_hz = (
        mids.target_fc_hz
        if target_fc_hz is None and mids is not None
        else target_fc_hz
    )
    target_fc_hz = _require_finite_positive("target_fc_hz", target_fc_hz)
    A_one = entry_area_cm2 * 1e-4
    A = A_one * port_count
    A_exit_one = chamber_area_cm2 * 1e-4
    L_geom = tube_depth_mm * 1e-3
    if chamber_volume_cc is None and mids is not None:
        chamber_volume_cc = mids.chamber.resolved_volume_cc(
            mids.driver,
            mids.port,
            mids.target_fc_hz,
        )
    elif chamber_volume_cc is None:
        if port_model == "uniform":
            L_default = L_geom + end_correction(
                A,
                end_corr,
                n_parallel=port_count,
            )
            denom_default = L_default / A
        elif port_model == "frustum":
            # Ingard depends on the still-unknown chamber volume, so use the
            # matching Rayleigh frustum as the closed-form sizing seed.
            a_entry = math.sqrt(A_one / math.pi)
            a_exit = math.sqrt(A_exit_one / math.pi)
            denom_one, _, _ = frustum_port_inertance_denominator(
                a_entry,
                a_exit,
                L_geom,
                end_corr=end_corr,
            )
            denom_default = denom_one / port_count
        else:
            raise ValueError(f"unknown port_model {port_model!r}")
        omega_over_c = (2.0 * math.pi * target_fc_hz) / c
        chamber_volume_cc = 1.0 / (
            denom_default * omega_over_c ** 2
        ) * 1e6
    else:
        chamber_volume_cc = _require_finite_positive(
            "chamber_volume_cc", chamber_volume_cc,
        )

    V = chamber_volume_cc * 1e-6
    entry_delta = 0.0
    exit_delta = 0.0
    if port_model == "uniform":
        denom, entry_delta, exit_delta, L_eff = _uniform_port_terms(
            A,
            L_geom,
            end_corr,
            port_count,
            chamber_volume_m3=V,
            interior_end_correction=interior_end_correction,
        )
        delta = entry_delta + exit_delta
        f = _helmholtz_from_inertance_denominator(V, denom, c)
    elif port_model == "frustum":
        a_entry = math.sqrt(A_one / math.pi)
        a_exit = math.sqrt(A_exit_one / math.pi)
        denom_one, entry_delta, exit_delta = frustum_port_inertance_denominator(
            a_entry,
            a_exit,
            L_geom,
            end_corr=end_corr,
            interior_end_correction=interior_end_correction,
            chamber_volume_m3=V,
        )
        denom = denom_one / port_count
        f = _helmholtz_from_inertance_denominator(V, denom, c)
        L_eff = denom * A
        delta = L_eff - L_geom
    else:
        raise ValueError(f"unknown port_model {port_model!r}")
    q_eval_hz = f if loss_eval_frequency_hz is None else _require_finite_positive(
        "loss_eval_frequency_hz", loss_eval_frequency_hz,
    )
    if loss_hydraulic_radius_m is None:
        loss_hydraulic_radius_m = math.sqrt(A_one / math.pi)
        if L_geom > 0.0:
            loss_hydraulic_radius_m = min(loss_hydraulic_radius_m, L_geom / 8.0)
    else:
        loss_hydraulic_radius_m = _require_finite_positive(
            "loss_hydraulic_radius_m", loss_hydraulic_radius_m,
        )
    q_port = viscothermal_port_q(
        q_eval_hz,
        A,
        hydraulic_radius_m=loss_hydraulic_radius_m,
        n_parallel=port_count,
        rho=rho,
    )

    cylinder_radius = None
    cylinder_depth = None
    chamber_shape = "explicit"
    if mids is not None:
        chamber_shape = mids.chamber.shape
    if mids is not None and mids.chamber.shape == "cylinder":
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
        "entry_delta_L_m": entry_delta,
        "interior_delta_L_m": exit_delta,
        "inertance_denominator_m_inv": denom,
        "f_Hz": f,
        "port_model": port_model,
        "interior_end_correction": interior_end_correction,
        "Q_port_derived": q_port,
        "Q_port_eval_hz": q_eval_hz,
        "Q_port_hydraulic_radius_m": loss_hydraulic_radius_m,
        "Q_port_loss_model": "kirchhoff_benade",
        "target_fc_hz": target_fc_hz,
        "entry_area_cm2": entry_area_cm2,
        "entry_total_area_cm2": entry_area_cm2 * port_count,
        "chamber_area_cm2": chamber_area_cm2,
        "flare_ratio": chamber_area_cm2 / entry_area_cm2,
        "tube_depth_mm": tube_depth_mm,
        "chamber_shape": chamber_shape,
        "cylinder_radius_mm": cylinder_radius,
        "cylinder_depth_mm": cylinder_depth,
        "explicit_mid_geometry": explicit_mid,
        "formula_bias_note": (
            "research notes §2a/§2d: archived build-thread measurements / "
            "Ingard 1953 indicate "
            "the closed form over-shoots the V→f_H shift by ~1.75x; "
            "BEM/measurement should set final tuning"
        ),
    }


def mid_chamber_helmholtz_from_params(
    params: "CabinetParams",
    **kwargs,
) -> Dict[str, object]:
    """Compute one mid-chamber Helmholtz estimate from a full cabinet
    params object (uses ``params.mids``; duck-typed, see module docstring)."""
    return mid_chamber_helmholtz(params.mids, **kwargs)


# ── Deprecated provenance aliases ─────────────────────────────────────────


def _deprecated_alias(replacement, old_name: str):
    """Wrap *replacement* under its historical project-prefixed name."""

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated; use "
            f"hornlab_sim.methods.helmholtz.{replacement.__name__}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = (
        f"Deprecated provenance alias for :func:`{replacement.__name__}`."
    )
    return wrapper


bigmeh_slot_helmholtz = _deprecated_alias(
    slot_helmholtz, "bigmeh_slot_helmholtz")
bigmeh_slot_helmholtz_from_params = _deprecated_alias(
    slot_helmholtz_from_params, "bigmeh_slot_helmholtz_from_params")
bigmeh_mid_chamber_helmholtz = _deprecated_alias(
    mid_chamber_helmholtz, "bigmeh_mid_chamber_helmholtz")
bigmeh_mid_chamber_helmholtz_from_params = _deprecated_alias(
    mid_chamber_helmholtz_from_params, "bigmeh_mid_chamber_helmholtz_from_params")


def main() -> None:
    """CLI: print a comparison table for slot widths."""
    import argparse
    p = argparse.ArgumentParser(prog="hornlab_sim.methods.helmholtz")
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
        r = mid_chamber_helmholtz(end_corr=args.end_corr)
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
            r = slot_helmholtz(
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
