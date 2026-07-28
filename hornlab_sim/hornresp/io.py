"""Parse Hornresp export files (.txt config + tab-separated response data).

Hornresp config format example (relevant lines for bandpass):

    Sd = 522.00          # cm^2
    Bl = 11.86           # T·m
    Cms = 1.92E-04       # m/N
    Rms = 1.52           # N·s/m
    Mmd = 36.80          # g  (mass without air load)
    Le  = 0.54           # mH
    Re  = 5.60           # ohm
    BP6S = 1             # 6th-order bandpass series flag

    Vc1 = 10.00          # L   (front chamber volume)
    Lc1 = 4.00           # cm  (front chamber depth — informational)
    Ap1 = 142.80         # cm^2 (front port area)
    Lp1 = 0.01           # cm   (front port length)
    Vc2 = 60.00          # L   (rear chamber volume)
    Lc2 = 60.00          # cm
    Ap2 = 80.00          # cm^2
    Lp2 = 14.20          # cm

    Ang = 2.0 x Pi       # half-space radiation
    Eg  = 2.00           # input voltage
    Rg  = 0.00

Hornresp response file: three tab-separated columns
    Freq (Hz)   SPL (dB)   WPhase (deg)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HornrespConfig:
    """Subset of Hornresp parameters relevant to lumped BP modelling."""
    Sd: float           # m^2
    Bl: float
    Re: float
    Le: float           # H
    Mmd: float          # kg
    Cms: float
    Rms: float
    Vc1: float          # m^3 (front / closer to driver)
    Ap1: float          # m^2
    Lp1: float          # m
    Vc2: float          # m^3 (rear)
    Ap2: float          # m^2
    Lp2: float          # m
    Eg: float           # V
    Rg: float
    Ang_steradians: float
    BP6S: bool
    BP4: bool
    n_drivers: int      # parsed from "BP4 = NP/NS" or "BP6S = NP/NS"
    wiring: str         # "P" (parallel), "S" (series), or "" (single)
    raw: dict[str, str]

    @property
    def half_space(self) -> bool:
        return abs(self.Ang_steradians - 2 * np.pi) < 1e-3


def _parse_topology_flag(s: str) -> tuple[int, str]:
    """Parse Hornresp's BP4/BP6S flag into (n_drivers, wiring).

    Hornresp encodes the driver count and wiring directly in the flag:
        BP4 = 0      → topology not active
        BP4 = 1      → single driver
        BP4 = 2P     → 2 drivers, parallel-wired
        BP4 = 2S     → 2 drivers, series-wired
        BP4 = 4P     → 4 drivers, parallel
        ... and same for BP6S.
    """
    s = s.strip().upper()
    if not s:
        return 0, ""
    m = re.match(r"^(\d+)\s*([PS]?)\s*$", s)
    if not m:
        return 0, ""
    n = int(m.group(1))
    wiring = m.group(2) or ""
    if n <= 1:
        return n, ""
    return n, wiring or "P"   # default >1 to parallel if not specified


def _parse_value(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    # Handle formats like "2.0 x Pi"
    if "Pi" in s or "pi" in s:
        m = re.match(r"([\d.+\-eE]+)\s*x\s*Pi", s)
        if m:
            return float(m.group(1)) * float(np.pi)
    # plain number
    try:
        return float(s)
    except ValueError:
        return None


def parse_config(path: Path | str) -> HornrespConfig:
    """Parse only the TRADITIONAL DRIVER + HORN sections.

    Hornresp exports include "ADVANCED" sections that re-declare keys like
    Le, Rms, Cms with zeroed values when those models are off. Track the
    active section header (lines starting with "|") and ignore advanced
    blocks unless they are actually enabled (Status Flags govern that).
    """
    SECTIONS_TO_USE = {
        "RADIATION, SOURCE AND MOUTH PARAMETER VALUES",
        "HORN PARAMETER VALUES",
        "TRADITIONAL DRIVER PARAMETER VALUES",
        "MAXIMUM SPL PARAMETER VALUES",
    }
    raw: dict[str, str] = {}
    text = Path(path).read_text(errors="replace")
    section: str = ""
    use_section = True
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("|"):
            section = stripped.lstrip("|").rstrip(":").strip()
            use_section = section in SECTIONS_TO_USE
            continue
        if not use_section:
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key:
            raw[key] = val

    def f(k: str) -> Optional[float]:
        return _parse_value(raw.get(k, ""))

    Sd_cm2 = f("Sd") or 0.0
    Vc1_L = f("Vc1") or 0.0
    Vc2_L = f("Vc2") or 0.0
    Ap1_cm2 = f("Ap1") or 0.0
    Ap2_cm2 = f("Ap2") or 0.0
    Lp1_cm = f("Lp1") or 0.0
    Lp2_cm = f("Lp2") or 0.0
    Mmd_g = f("Mmd") or 0.0
    Le_mH = f("Le") or 0.0
    Ang = _parse_value(raw.get("Ang", ""))
    if Ang is None:
        Ang = 2 * np.pi
    Eg = f("Eg")
    if Eg is None:
        Eg = 2.83

    bp6s_n, bp6s_w = _parse_topology_flag(raw.get("BP6S", ""))
    bp4_n, bp4_w = _parse_topology_flag(raw.get("BP4", ""))
    if bp6s_n > 0:
        is_bp6s, is_bp4 = True, False
        n_drivers, wiring = bp6s_n, bp6s_w
    elif bp4_n > 0:
        is_bp6s, is_bp4 = False, True
        n_drivers, wiring = bp4_n, bp4_w
    else:
        is_bp6s, is_bp4 = False, False
        n_drivers, wiring = 1, ""

    return HornrespConfig(
        Sd=Sd_cm2 * 1e-4,
        Bl=f("Bl") or 0.0,
        Re=f("Re") or 0.0,
        Le=Le_mH * 1e-3,
        Mmd=Mmd_g * 1e-3,
        Cms=f("Cms") or 0.0,
        Rms=f("Rms") or 0.0,
        Vc1=Vc1_L * 1e-3,
        Ap1=Ap1_cm2 * 1e-4,
        Lp1=Lp1_cm * 1e-2,
        Vc2=Vc2_L * 1e-3,
        Ap2=Ap2_cm2 * 1e-4,
        Lp2=Lp2_cm * 1e-2,
        Eg=Eg,
        Rg=f("Rg") or 0.0,
        Ang_steradians=Ang,
        BP6S=is_bp6s,
        BP4=is_bp4,
        n_drivers=n_drivers,
        wiring=wiring,
        raw=raw,
    )


def parse_response(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freq, spl_db, phase_deg)."""
    rows = []
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("freq"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float).reshape(-1, 3)
    return arr[:, 0], arr[:, 1], arr[:, 2]
