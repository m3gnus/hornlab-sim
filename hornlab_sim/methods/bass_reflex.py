"""WinISD-style bass-reflex seed screen with explicit short-port scoring.

This is a fast LEM screening tool, not a final enclosure optimizer. It uses
WinISD-like TS heuristics for the starting advice:

* EBP = Fs / Qes for sealed-vs-vented hinting,
* sealed Qtc=0.707 starter volume,
* a practical vented sweep over Vb, Fb, and physical port length.

The vented sweep is deliberately port-geometry-aware. For each Vb/Fb it samples
short physical port lengths, derives the required area, rejects candidates whose
port air velocity would limit output before excursion/thermal limits, and
reports the first longitudinal port resonance. Shorter feasible ports score
better when output and ripple are comparable.

Example:
    from hornlab_sim.methods.bass_reflex import SweepConfig, run_screen
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from hornlab_sim.methods.bandpass import C_SOUND, Chamber, Driver, Port, simulate

V_REF = 2.83
FRONT_FREE_AIR_VOLUME_M3 = 1.0e6
PORT_FLANGED_INSIDE = False
PORT_FLANGED_OUTSIDE = True
PORT_END_CORR_FACTOR = 0.61 + 0.85
DEFAULT_QTC = 0.707


@dataclass(frozen=True)
class SweepConfig:
    size_in: float | None
    vb_values_l: np.ndarray
    fb_values_hz: np.ndarray
    port_lengths_m: np.ndarray
    score_band: tuple[float, float]
    velocity_cap_mps: float
    max_equiv_diam_mm: float
    min_port_resonance_hz: float
    top_n: int


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def sval(row: dict[str, str], key: str) -> str:
    return row.get(key, "") or ""


def frange(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("range step must be positive")
    count = int(math.floor((stop - start) / step + 0.5)) + 1
    return start + step * np.arange(max(count, 0), dtype=float)


def load_rows(path: Path, size_in: float | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if size_in is not None and abs(fval(row, "Size_in", -999.0) - size_in) > 1.0e-9:
                continue
            rows.append(row)
    return rows


def row_to_driver(row: dict[str, str]) -> Driver | None:
    sd_m2 = fval(row, "Sd_cm2") * 1e-4
    bl = fval(row, "Bl_Tm")
    re = fval(row, "Re_ohm")
    le = fval(row, "Le_mH") * 1e-3
    mms = fval(row, "Mms_g") * 1e-3
    fs = fval(row, "Fs_Hz")
    qms = fval(row, "Qms", 5.0) or 5.0
    if min(sd_m2, bl, re, mms, fs, qms) <= 0:
        return None
    cms = 1.0 / ((2.0 * math.pi * fs) ** 2 * mms)
    rms = (2.0 * math.pi * fs * mms) / qms
    return Driver(Sd=sd_m2, Bl=bl, Re=re, Le=le, Mms=mms, Cms=cms, Rms=rms)


def ebp_hint(fs_hz: float, qes: float) -> tuple[float | None, str]:
    if fs_hz <= 0 or qes <= 0:
        return None, "unknown"
    ebp = fs_hz / qes
    if 45.0 <= ebp <= 65.0:
        return ebp, "borderline"
    if ebp < 55.0:
        return ebp, "sealed"
    return ebp, "vented"


def sealed_qtc_seed(row: dict[str, str], qtc: float = DEFAULT_QTC) -> dict[str, object]:
    fs_hz = fval(row, "Fs_Hz")
    qts = fval(row, "Qts")
    qes = fval(row, "Qes")
    vas_l = fval(row, "Vas_L")
    ebp, hint = ebp_hint(fs_hz, qes)
    out: dict[str, object] = {
        "Brand": sval(row, "Brand"),
        "Model": sval(row, "Model"),
        "Z_ohm": fval(row, "Z_ohm"),
        "Size_in": fval(row, "Size_in"),
        "Fs_Hz": fs_hz,
        "Qts": qts,
        "Qes": qes,
        "Vas_L": vas_l,
        "EBP": ebp,
        "EBP_hint": hint,
        "Qtc_target": qtc,
    }
    if min(fs_hz, qts, vas_l) <= 0:
        out["Sealed_status"] = "missing_required_ts"
        return out
    if qts >= qtc:
        out["Sealed_status"] = "qts_at_or_above_qtc_target"
        return out
    ratio = qtc / qts
    vb_l = vas_l / (ratio * ratio - 1.0)
    out.update(
        {
            "Sealed_status": "ok",
            "Sealed_Vb_L": vb_l,
            "Sealed_Fc_Hz": fs_hz * ratio,
        }
    )
    return out


def port_area_for_length_m(vb_l: float, fb_hz: float, lp_m: float) -> float | None:
    """Required round-port-equivalent area for Vb/Fb and physical length."""

    vb_m3 = vb_l * 1e-3
    if min(vb_m3, fb_hz) <= 0 or lp_m < 0:
        return None
    k2 = (2.0 * math.pi * fb_hz / C_SOUND) ** 2
    a = 1.0 / (vb_m3 * k2)
    b = PORT_END_CORR_FACTOR / math.sqrt(math.pi)
    disc = b * b + 4.0 * a * lp_m
    if disc < 0:
        return None
    x = (b + math.sqrt(disc)) / (2.0 * a)
    area = x * x
    return area if area > 0 else None


def effective_port_length_m(area_m2: float, lp_m: float) -> float:
    return lp_m + PORT_END_CORR_FACTOR * math.sqrt(area_m2 / math.pi)


def passband_knee(
    freq: np.ndarray,
    spl: np.ndarray,
    ref_db: float,
    drop_db: float,
    side: str,
    peak_idx: int,
) -> float | None:
    target = ref_db - drop_db
    if side == "low":
        previous = peak_idx
        for idx in range(peak_idx, -1, -1):
            if spl[idx] <= target:
                if idx == peak_idx:
                    return float(freq[idx])
                denom = spl[previous] - spl[idx]
                if abs(denom) < 1.0e-12:
                    return float(freq[idx])
                frac = (spl[previous] - target) / denom
                return float(
                    np.exp(np.log(freq[previous]) + frac * (np.log(freq[idx]) - np.log(freq[previous])))
                )
            previous = idx
        return None

    previous = peak_idx
    for idx in range(peak_idx, len(freq)):
        if spl[idx] <= target:
            if idx == peak_idx:
                return float(freq[idx])
            denom = spl[previous] - spl[idx]
            if abs(denom) < 1.0e-12:
                return float(freq[idx])
            frac = (spl[previous] - target) / denom
            return float(
                np.exp(np.log(freq[previous]) + frac * (np.log(freq[idx]) - np.log(freq[previous])))
            )
        previous = idx
    return None


def price_eur(row: dict[str, str]) -> float | None:
    for key in ("Price_min_EUR", "Price_EUR"):
        value = fval(row, key, 0.0)
        if value > 0:
            return value
    return None


def alignment_metrics(
    row: dict[str, str],
    driver: Driver,
    freq: np.ndarray,
    config: SweepConfig,
    vb_l: float,
    fb_hz: float,
    port_length_m: float,
) -> dict[str, object] | None:
    port_area_m2 = port_area_for_length_m(vb_l, fb_hz, port_length_m)
    if port_area_m2 is None:
        return None
    equiv_diam_mm = 2.0 * math.sqrt(port_area_m2 / math.pi) * 1000.0
    if equiv_diam_mm > config.max_equiv_diam_mm:
        return None

    port = Port(
        area=port_area_m2,
        length=port_length_m,
        flanged_inside=PORT_FLANGED_INSIDE,
        flanged_outside=PORT_FLANGED_OUTSIDE,
        n_parallel=1,
    )
    try:
        sim = simulate(
            driver=driver,
            front_chamber=Chamber(volume=FRONT_FREE_AIR_VOLUME_M3),
            rear_chamber=Chamber(volume=vb_l * 1e-3, port=port),
            freq=freq,
            v_g=V_REF,
            driver_radiates_directly=True,
        )
    except Exception:
        return None

    xmax_mm = fval(row, "Xmax_mm")
    re_ohm = fval(row, "Re_ohm")
    pmax_w = fval(row, "Power_W")
    if min(xmax_mm, re_ohm, pmax_w) <= 0:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        v_xmax = np.where(sim.cone_excursion_mm > 1.0e-12, xmax_mm / sim.cone_excursion_mm * V_REF, np.inf)
        v_pmax = math.sqrt(pmax_w * re_ohm)
        v_no_port_limit = np.minimum(v_xmax, v_pmax)
        u_port = np.abs(sim.U_rear_port) if sim.U_rear_port is not None else np.zeros_like(freq)
        v_port_limit = np.where(u_port > 1.0e-15, config.velocity_cap_mps * port_area_m2 / u_port * V_REF, np.inf)

    band_mask = (freq >= config.score_band[0]) & (freq <= config.score_band[1])
    if not np.any(band_mask):
        return None
    port_margin = float(np.min(v_port_limit[band_mask] / v_no_port_limit[band_mask]))
    if port_margin < 1.0:
        return None

    spl_limit = sim.spl_total + 20.0 * np.log10(v_no_port_limit / V_REF)
    band_idxs = np.where(band_mask)[0]
    spl_band = spl_limit[band_mask]
    idx_min = int(band_idxs[np.argmin(spl_band)])
    idx_max = int(band_idxs[np.argmax(spl_band)])
    min_band = float(np.min(spl_band))
    max_band = float(np.max(spl_band))
    ripple = max_band - min_band
    avg_band = float(np.mean(spl_band))

    leff_m = effective_port_length_m(port_area_m2, port_length_m)
    port_res_hz = C_SOUND / (2.0 * leff_m) if leff_m > 1.0e-9 else math.inf
    if port_res_hz < config.min_port_resonance_hz:
        return None
    res_margin = port_res_hz / config.score_band[1]
    short_score = (
        min_band
        - 0.45 * ripple
        + 2.0 * math.log2(max(res_margin, 1.0))
        - 0.010 * vb_l
        - 0.006 * (port_length_m * 1000.0)
    )
    ebp, hint = ebp_hint(fval(row, "Fs_Hz"), fval(row, "Qes"))

    result: dict[str, object] = {
        "Brand": sval(row, "Brand"),
        "Model": sval(row, "Model"),
        "Z_ohm": fval(row, "Z_ohm"),
        "Size_in": fval(row, "Size_in"),
        "Price_EUR": price_eur(row),
        "EBP": ebp,
        "EBP_hint": hint,
        "Fb_Hz": fb_hz,
        "Vb_net_L": vb_l,
        "Port_Lp_mm": port_length_m * 1000.0,
        "Port_L_eff_mm": leff_m * 1000.0,
        "Port_area_cm2": port_area_m2 * 1e4,
        "Port_diam_equiv_mm": equiv_diam_mm,
        "Port_volume_L": port_area_m2 * port_length_m * 1000.0,
        "Port_first_resonance_Hz": port_res_hz,
        "Port_resonance_to_band_high": res_margin,
        "Port_velocity_margin": port_margin,
        "MinSPL_band_dB": min_band,
        "MaxSPL_band_dB": max_band,
        "AvgSPL_band_dB": avg_band,
        "Ripple_band_dB": ripple,
        "F_at_MinSPL_Hz": float(freq[idx_min]),
        "F_at_MaxSPL_Hz": float(freq[idx_max]),
        "F3_low_Hz": passband_knee(freq, spl_limit, max_band, 3.0, "low", idx_max),
        "ShortPortScore": short_score,
        "Fs_Hz": fval(row, "Fs_Hz"),
        "Qts": fval(row, "Qts"),
        "Qes": fval(row, "Qes"),
        "Qms": fval(row, "Qms"),
        "Vas_L": fval(row, "Vas_L"),
        "Sd_cm2": fval(row, "Sd_cm2"),
        "Bl_Tm": fval(row, "Bl_Tm"),
        "Re_ohm": re_ohm,
        "Mms_g": fval(row, "Mms_g"),
        "Xmax_mm": xmax_mm,
        "Pmax_W": pmax_w,
        "Sensitivity_dB": fval(row, "Sensitivity_dB"),
        "Weight_kg": fval(row, "Weight_kg"),
        "Notes": sval(row, "Notes"),
    }
    for sample_hz in (25, 28, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200):
        if freq[0] <= sample_hz <= freq[-1]:
            result[f"SPL_at_{sample_hz}Hz_dB"] = float(np.interp(float(sample_hz), freq, spl_limit))
    return result


def best_by_driver(rows: Iterable[dict[str, object]], key_field: str) -> list[dict[str, object]]:
    best: dict[tuple[str, str, float], dict[str, object]] = {}
    for row in rows:
        key = (str(row["Brand"]), str(row["Model"]), float(row["Z_ohm"]))
        current = best.get(key)
        if current is None or float(row[key_field]) > float(current[key_field]):
            best[key] = row
    return list(best.values())


def fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.3f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


FIELDS = [
    "Rank",
    "Brand",
    "Model",
    "Z_ohm",
    "Price_EUR",
    "EBP",
    "EBP_hint",
    "ShortPortScore",
    "MinSPL_band_dB",
    "AvgSPL_band_dB",
    "MaxSPL_band_dB",
    "Ripple_band_dB",
    "F_at_MinSPL_Hz",
    "F3_low_Hz",
    "Fb_Hz",
    "Vb_net_L",
    "Port_Lp_mm",
    "Port_L_eff_mm",
    "Port_first_resonance_Hz",
    "Port_resonance_to_band_high",
    "Port_velocity_margin",
    "Port_area_cm2",
    "Port_diam_equiv_mm",
    "Port_volume_L",
    "SPL_at_25Hz_dB",
    "SPL_at_28Hz_dB",
    "SPL_at_30Hz_dB",
    "SPL_at_35Hz_dB",
    "SPL_at_40Hz_dB",
    "SPL_at_50Hz_dB",
    "SPL_at_60Hz_dB",
    "SPL_at_80Hz_dB",
    "SPL_at_100Hz_dB",
    "SPL_at_120Hz_dB",
    "SPL_at_150Hz_dB",
    "SPL_at_200Hz_dB",
    "Fs_Hz",
    "Qts",
    "Qes",
    "Qms",
    "Vas_L",
    "Sd_cm2",
    "Bl_Tm",
    "Re_ohm",
    "Mms_g",
    "Xmax_mm",
    "Pmax_W",
    "Sensitivity_dB",
    "Weight_kg",
    "Notes",
]

SEALED_FIELDS = [
    "Brand",
    "Model",
    "Z_ohm",
    "Size_in",
    "Fs_Hz",
    "Qts",
    "Qes",
    "Vas_L",
    "EBP",
    "EBP_hint",
    "Qtc_target",
    "Sealed_status",
    "Sealed_Vb_L",
    "Sealed_Fc_Hz",
]


def add_rank(rows: list[dict[str, object]], sort_field: str) -> list[dict[str, object]]:
    out = [dict(row) for row in sorted(rows, key=lambda row: float(row[sort_field]), reverse=True)]
    for idx, row in enumerate(out, start=1):
        row["Rank"] = idx
    return out


def plot_best(path: Path, rows: list[dict[str, object]], band: tuple[float, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    top = rows[: min(20, len(rows))]
    if not top:
        return
    labels = [f"{row['Brand']} {row['Model']}"[:34] for row in top]
    scores = [float(row["ShortPortScore"]) for row in top]
    resonances = [float(row["Port_first_resonance_Hz"]) for row in top]
    y = np.arange(len(top))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.34 * len(top))))
    axes[0].barh(y, scores, color="#2a6f97")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Short-port score")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(y, resonances, color="#c17c3a")
    axes[1].axvline(band[1], color="0.25", lw=1.0, ls="--", label="band high")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("First port resonance (Hz)")
    axes[1].grid(True, axis="x", alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Bass-reflex short-port screen: best per driver", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_screen(rows: list[dict[str, str]], config: SweepConfig, output_dir: Path) -> None:
    freq = np.logspace(
        math.log10(max(10.0, config.score_band[0] * 0.5)),
        math.log10(config.score_band[1] * 2.5),
        360,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sealed_rows = [sealed_qtc_seed(row) for row in rows]
    write_csv(output_dir / "sealed_qtc_hints.csv", sealed_rows, SEALED_FIELDS)

    feasible: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in rows:
        driver = row_to_driver(row)
        if driver is None:
            skipped.append({"Brand": sval(row, "Brand"), "Model": sval(row, "Model"), "Reason": "missing_required_ts"})
            continue
        start_count = len(feasible)
        for vb_l in config.vb_values_l:
            for fb_hz in config.fb_values_hz:
                for lp_m in config.port_lengths_m:
                    metrics = alignment_metrics(row, driver, freq, config, float(vb_l), float(fb_hz), float(lp_m))
                    if metrics is not None:
                        feasible.append(metrics)
        if len(feasible) == start_count:
            skipped.append({"Brand": sval(row, "Brand"), "Model": sval(row, "Model"), "Reason": "no_feasible_short_port_alignment"})

    all_by_score = add_rank(feasible, "ShortPortScore")
    best_score = add_rank(best_by_driver(feasible, "ShortPortScore"), "ShortPortScore")
    best_min_spl = add_rank(best_by_driver(feasible, "MinSPL_band_dB"), "MinSPL_band_dB")
    write_csv(output_dir / "all_feasible_alignments.csv", all_by_score, FIELDS)
    write_csv(output_dir / "best_short_port_by_driver.csv", best_score, FIELDS)
    write_csv(output_dir / "best_min_spl_by_driver.csv", best_min_spl, FIELDS)
    write_csv(output_dir / "skipped_rows.csv", skipped, ["Brand", "Model", "Reason"])
    plot_best(output_dir / "best_short_port_by_driver.png", best_score, config.score_band)

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Bass-reflex short-port screen\n\n"
            "Fast LEM screen using WinISD-style EBP/Qtc hints plus a practical "
            "vented sweep that prefers short, high-resonance ports.\n\n"
            f"- Drivers loaded: {len(rows)}\n"
            f"- Feasible alignments: {len(feasible)}\n"
            f"- Drivers with feasible alignments: {len(best_score)}\n"
            f"- Score band: {config.score_band[0]:.0f}-{config.score_band[1]:.0f} Hz\n"
            f"- Port velocity cap: {config.velocity_cap_mps:.1f} m/s\n"
            f"- Port length sweep: {config.port_lengths_m[0] * 1000:.0f}-"
            f"{config.port_lengths_m[-1] * 1000:.0f} mm\n"
            f"- Minimum accepted first port resonance: {config.min_port_resonance_hz:.0f} Hz\n\n"
            "The first port resonance is estimated as `c / (2 * L_eff)` using the "
            "same end-correction convention as the Helmholtz sizing step. This is "
            "a screening penalty, not a replacement for later BEM/TMM validation.\n"
        )

    print(f"loaded rows: {len(rows)}")
    print(f"feasible alignments: {len(feasible)}")
    print(f"drivers with feasible alignments: {len(best_score)}")
    print(f"output: {output_dir}")
    if best_score:
        print("\nTop short-port candidates:")
        for row in best_score[: config.top_n]:
            print(
                f"{int(row['Rank']):2d}. {row['Brand']} {row['Model']} "
                f"score={float(row['ShortPortScore']):.1f} "
                f"min={float(row['MinSPL_band_dB']):.1f} dB "
                f"Vb={float(row['Vb_net_L']):.0f} L Fb={float(row['Fb_Hz']):.0f} Hz "
                f"Lp={float(row['Port_Lp_mm']):.0f} mm "
                f"fport={float(row['Port_first_resonance_Hz']):.0f} Hz"
            )


__all__ = [
    "DEFAULT_QTC",
    "FIELDS",
    "FRONT_FREE_AIR_VOLUME_M3",
    "PORT_END_CORR_FACTOR",
    "PORT_FLANGED_INSIDE",
    "PORT_FLANGED_OUTSIDE",
    "SEALED_FIELDS",
    "SweepConfig",
    "V_REF",
    "add_rank",
    "alignment_metrics",
    "best_by_driver",
    "ebp_hint",
    "effective_port_length_m",
    "fmt",
    "frange",
    "fval",
    "load_rows",
    "passband_knee",
    "plot_best",
    "port_area_for_length_m",
    "price_eur",
    "row_to_driver",
    "run_screen",
    "sealed_qtc_seed",
    "sval",
    "write_csv",
]
