"""Export bandpass designs to Hornresp .txt config format.

Produces files importable by Hornresp directly (round-tripped against an
actual Hornresp export from the user's system, 2026-05-03).

Critical Hornresp file-format rules learned the hard way:

* **Encoding & line endings**: ISO-8859-1, CRLF (`\\r\\n`). Hornresp is a
  Windows app — UTF-8 + LF will fail the import.
* **Section structure**: 14 named sections starting with `|HEADER:`, each
  followed by a blank line. The order matters; replicate it exactly.
* **Topology flag** lives in the |TRADITIONAL DRIVER PARAMETER VALUES|
  section as a single line:
      `BP4 = N{P|S}`   for 4th-order bandpass, sealed rear + ported front
      `BP6S = N{P|S}`  for 6th-order series bandpass, both ported
  where N is the number of drivers and the suffix is **P** (parallel) or
  **S** (series). Examples that Hornresp accepts:
      BP4 = 1          (one driver — suffix omitted)
      BP4 = 2P         (two drivers parallel-wired)
      BP6S = 4S        (four drivers series-wired)
  **Driver TS values are PER DRIVER** even when N>1. Hornresp scales
  internally based on N and the wiring. Do NOT pre-scale Sd/Re/Mms
  yourself — that breaks the import.
* **`L23` and `F23`** are required between Lp2 and S3 in newer Hornresp
  versions (segment-2-to-3 length and flare); set to 0 when unused.
* **`Added Mass = 0.00`** (not empty string).
* Numeric formatting: 4 decimals for chamber volumes / port dimensions
  so small values like 0.0363 L don't get rounded to 0.04.

Topology mapping for the |HORN PARAMETER VALUES| section:

  BP4 (sealed rear + ported front) — output via front port
      Vc1, Ap1, Lp1  : front chamber + front port
      Vc2            : sealed rear volume (Ap2 = Lp2 = 0)
      BP4 = N{P|S}

  BP6S (dual-tuned bandpass) — output via both ports
      Vc1, Ap1, Lp1  : front chamber + front port
      Vc2, Ap2, Lp2  : rear chamber + rear port
      BP6S = N{P|S}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hornlab_sim.methods.bandpass import Chamber, Driver, Port


@dataclass
class HornrespExport:
    driver: Driver
    front_chamber: Chamber           # ported (front_chamber.port required)
    rear_chamber: Chamber            # ported for BP6S, sealed for BP4
    Eg: float = 2.83                 # input voltage (V)
    Rg: float = 0.0                  # source resistance (Ω)
    half_space: bool = True          # Hornresp Ang = 2π (vs 4π full space)
    comment: str = ""
    Pamp: int = 100
    Vamp: int = 25
    Iamp: int = 4
    Pmax: int = 500
    Xmax_mm: float = 5.0
    Lc1_cm: float = 4.0              # informational; Hornresp displays it
    Lc2_cm: float = 0.0              # informational
    wiring: str = "P"                # "P" parallel or "S" series, when n>1

    def render(self) -> str:
        d = self.driver.derive()
        f = self.front_chamber
        r = self.rear_chamber

        # Convert SI → Hornresp units (cm, L, g, mH).
        Sd_cm2 = d.Sd * 1e4
        Mmd_g = (d.Mmd if d.Mmd is not None else d.Mms) * 1e3
        Le_mH = d.Le * 1e3

        Vc1_L = f.volume * 1e3
        Vc2_L = r.volume * 1e3
        Ap1_cm2 = (f.port.area if f.port else 0.0) * 1e4
        Lp1_cm = (f.port.length if f.port else 0.0) * 1e2
        Ap2_cm2 = (r.port.area if r.port else 0.0) * 1e4
        Lp2_cm = (r.port.length if r.port else 0.0) * 1e2

        is_bp6s = (r.port is not None) and (f.port is not None)
        is_bp4 = (f.port is not None) and (r.port is None)
        n = self.driver.n_drivers
        # Hornresp wants "1" for a single driver, "NP" / "NS" for N>1.
        if n <= 1:
            n_suffix = "1"
        else:
            w = self.wiring.upper() if self.wiring.upper() in ("P", "S") else "P"
            n_suffix = f"{n}{w}"

        ang = "2.0 x Pi" if self.half_space else "4.0 x Pi"

        # Use the section header order Hornresp produces. Numeric formats
        # mimic Hornresp's typical 2-decimal output.
        lines = []
        push = lines.append
        push(f"ID = 55.30")
        push("")
        push(f"Comment = {self.comment or 'Lumped BP export'}")
        push("")
        push("|RADIATION, SOURCE AND MOUTH PARAMETER VALUES:")
        push("")
        push(f"Ang = {ang}")
        push(f"Eg = {self.Eg:.2f}")
        push(f"Rg = {self.Rg:.2f}")
        push("Cir = 0.00")
        push("")
        push("|HORN PARAMETER VALUES:")
        push("")
        # Use 4 decimals for volumes/lengths so small values (e.g. 36 mL
        # front chamber, 1.5 cm port) survive round-trip through Hornresp.
        push(f"Vc1 = {Vc1_L:.4f}")
        push(f"Lc1 = {self.Lc1_cm:.2f}")
        push(f"Ap1 = {Ap1_cm2:.4f}")
        push(f"Lp1 = {Lp1_cm:.4f}")
        push(f"Vc2 = {Vc2_L:.4f}")
        push(f"Lc2 = {self.Lc2_cm:.2f}")
        push(f"Ap2 = {Ap2_cm2:.4f}")
        push(f"Lp2 = {Lp2_cm:.4f}")
        push("L23 = 0.00")
        push("F23 = 0.00")
        push("S3 = 0.00")
        push("Lo1 = 0.00")
        push("L34 = 0.00")
        push("F34 = 0.00")
        push("S4 = 0.00")
        push("Lo2 = 0.00")
        push("L45 = 0.00")
        push("F45 = 0.00")
        push("")
        push("|TRADITIONAL DRIVER PARAMETER VALUES:")
        push("")
        push(f"Sd = {Sd_cm2:.2f}")
        push(f"Bl = {d.Bl:.2f}")
        push(f"Cms = {d.Cms:.2E}")
        push(f"Rms = {d.Rms:.2f}")
        push(f"Mmd = {Mmd_g:.2f}")
        push(f"Le = {Le_mH:.2f}")
        push(f"Re = {d.Re:.2f}")
        if is_bp4:
            push(f"BP4 = {n_suffix}")
        elif is_bp6s:
            push(f"BP6S = {n_suffix}")
        else:
            push("BP6S = 0")
        push("")
        push("|ADVANCED DRIVER PARAMETER VALUES FOR SEMI-INDUCTANCE MODEL:")
        push("")
        push("Re' = 0.00")
        push("Leb = 0.00")
        push("Le = 0.00")
        push("Ke = 0.00")
        push("Rss = 0.00")
        push("")
        push("|ADVANCED DRIVER PARAMETER VALUES FOR FREQUENCY-DEPENDENT DAMPING MODEL:")
        push("")
        push("Rms = 0.00")
        push("Ams = 0.00")
        push("")
        push("|PASSIVE RADIATOR PARAMETER VALUE:")
        push("")
        push("Added Mass = 0.00")
        push("")
        push("|CHAMBER PARAMETER VALUES:")
        push("")
        push("Vrc = 0.00")
        push("Lrc = 0.00")
        push("Fr = 0.00")
        push("Tal = 0.00")
        push("Vtc = 0.00")
        push("Atc = 0.00")
        push("")
        push("Acoustic Path Length = 0.0")
        push("")
        push("|MAXIMUM SPL PARAMETER VALUES:")
        push("")
        push(f"Pamp = {self.Pamp}")
        push(f"Vamp = {self.Vamp}")
        push(f"Iamp = {self.Iamp}")
        push(f"Pmax = {self.Pmax}")
        push(f"Xmax = {self.Xmax_mm:.1f}")
        push("")
        push("Maximum SPL Setting = 3")
        push("")
        push("|ABSORBENT FILLING MATERIAL PARAMETER VALUES:")
        push("")
        push("Fr1 = 0.00")
        push("Fr2 = 0.00")
        push("Fr3 = 0.00")
        push("Fr4 = 0.00")
        push("")
        push("Tal1 = 100")
        push("Tal2 = 100")
        push("Tal3 = 100")
        push("Tal4 = 100")
        push("")
        push("|ACTIVE BAND PASS FILTER PARAMETER VALUES:")
        push("")
        push("High Pass Frequency = 0")
        push("High Pass Slope = 1")
        push("Low Pass Frequency = 0")
        push("Low Pass Slope = 1")
        push("")
        push("Butterworth High Pass Order = 1")
        push("Butterworth Low Pass Order = 1")
        push("Linkwitz-Riley High Pass Order = 2")
        push("Linkwitz-Riley Low Pass Order = 2")
        push("Bessel High Pass Order = 1")
        push("Bessel Low Pass Order = 1")
        push("")
        push("2nd Order High Pass Q = 0.5")
        push("2nd Order Low Pass Q = 0.5")
        push("4th Order High Pass Q = 0.5")
        push("4th Order Low Pass Q = 0.5")
        push("")
        push("Active Filter Alignment = 1")
        push("Active Filter On / Off Switch = 1")
        push("")
        push("|PASSIVE FILTER PARAMETER VALUES:")
        push("")
        push("Series / Parallel 1 = S")
        push("Series / Parallel 2 = S")
        push("Series / Parallel 3 = S")
        push("Series / Parallel 4 = S")
        push("")
        push("|EQUALISER FILTER PARAMETER VALUES:")
        push("")
        for i in range(1, 7):
            push(f"Band {i} Frequency = 0")
            push(f"Band {i} Q Factor = 0.01")
            push(f"Band {i} Gain = 0.0")
            push(f"Band {i} Type = -1")
        push("")
        push("|STATUS FLAGS:")
        push("")
        push("Auto Path Flag = 1")
        push("Lossy Inductance Model Flag = 0")
        push("Semi-Inductance Model Flag = 0")
        push("Damping Model Flag = 0")
        push("Closed Mouth Flag = 0")
        push("Continuous Flag = 1")
        push("End Correction Flag = 1")
        push("")
        push("|OTHER SETTINGS:")
        push("")
        push("Filter Type Index = 0")
        push("Filter Input Index = 0")
        push("Filter Output Index = 0")
        push("")
        push("Filter Type = 1")
        push("")
        push("MEH Configuration = 0")
        push("ME Amplifier Polarity Value = 1")
        # Hornresp is a Windows tool and expects CRLF line endings
        # plus ISO-8859-1 encoding. Use a literal "\r\n" join here so
        # the output matches a native Hornresp export byte-for-byte.
        return "\r\n".join(lines) + "\r\n"

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        # newline="" prevents Python from translating the embedded
        # "\r\n" into "\r\r\n" on Windows, and keeps LF-only systems
        # from stripping the CR.
        p.write_bytes(self.render().encode("iso-8859-1"))
        return p


def export_bp4(
    driver: Driver,
    Vb_front: float,
    front_port: Port,
    Vb_rear: float,
    path: Path | str,
    *,
    Eg: float = 2.83,
    comment: str = "BP4 export",
    **kwargs,
) -> Path:
    """Sealed rear + ported front. Vc2 carries the sealed rear volume,
    Ap2/Lp2 = 0 (Hornresp interprets this as a sealed chamber).
    """
    return HornrespExport(
        driver=driver,
        front_chamber=Chamber(volume=Vb_front, port=front_port),
        rear_chamber=Chamber(volume=Vb_rear, port=None),
        Eg=Eg,
        comment=comment,
        **kwargs,
    ).write(path)


def export_bp6s(
    driver: Driver,
    Vb_front: float, front_port: Port,
    Vb_rear: float, rear_port: Port,
    path: Path | str,
    *,
    Eg: float = 2.83,
    comment: str = "BP6S export",
    **kwargs,
) -> Path:
    return HornrespExport(
        driver=driver,
        front_chamber=Chamber(volume=Vb_front, port=front_port),
        rear_chamber=Chamber(volume=Vb_rear, port=rear_port),
        Eg=Eg,
        comment=comment,
        **kwargs,
    ).write(path)
