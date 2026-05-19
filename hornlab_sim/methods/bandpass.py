"""Lumped-parameter electroacoustic models for bandpass loudspeakers.

Computes on-axis frequency response, port volume velocities, cone
excursion, and electrical impedance for two bandpass topologies:

- BP4: 4th-order bandpass — sealed rear chamber + ported front chamber.
  Output exits via the front port only.
- BP6S: 6th-order bandpass — both chambers ported. Output exits via both
  ports (summed coherently, assuming coincident far-field source positions).

Convention: acoustic-domain equivalent circuit (pressure / volume velocity).
Driver Thevenin source p_g = Bl·v_g / (Sd·Z_e) drives the network through
acoustic source impedance Z_drv = Mas·s + Ras + 1/(s·Cas) + Bl²/(Sd²·Z_e).

This is the same lumped-parameter framework Hornresp uses below the lowest
1D standing-wave mode of the chambers/ports. Above those modes — in
chambers more than ~λ/4 long at the frequency of interest — Hornresp uses a
1D transmission-line model that captures organ-pipe resonances; this
module does NOT model those modes (so will under-predict the peaks/dips
above the chambers' lowest 1D mode).

Reference texts: Beranek, Acoustics; Small, "Direct-Radiator Loudspeaker
System Analysis" (JAES 1972).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .port_acoustics import end_correction as _pa_end_correction

RHO = 1.21          # air density (kg/m^3) at ~20 C
C_SOUND = 343.0     # speed of sound (m/s)
P_REF = 20e-6       # SPL reference pressure (Pa)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class Driver:
    """Thiele-Small parameters for a moving-coil driver.

    Provide either Mms (with air load) or Mmd (without). If only Mmd is
    given, free-air radiation mass is added on each face — but inside a
    small chamber the chamber compliance dominates over free-air mass
    loading so this approximation is fine.

    Either Cms, Vas, or Fs is required to specify suspension stiffness.
    Either Rms or Qms is required to specify mechanical damping (default
    Qms = 5 if neither given).

    n_drivers: parallel-wired identical drivers moving in unison
    (electrically and mechanically). The lumped network is solved as a
    single equivalent driver with Sd → n·Sd, Re → Re/n, etc.
    """
    Sd: float
    Bl: float
    Re: float
    Le: float = 0.0
    Mms: Optional[float] = None
    Mmd: Optional[float] = None
    Cms: Optional[float] = None
    Rms: Optional[float] = None
    Vas: Optional[float] = None
    Fs: Optional[float] = None
    Qms: Optional[float] = None
    n_drivers: int = 1

    def derive(self) -> "Driver":
        d = Driver(**self.__dict__)
        if d.Mms is None and d.Mmd is None:
            raise ValueError("Driver needs Mms or Mmd")
        if d.Mms is None:
            # In a bandpass enclosure both cone faces load against chamber
            # compliance, not free-air radiation mass. Treat Mms ≈ Mmd —
            # this matches Hornresp's chambered-driver behavior. For
            # open-baffle / unchambered cones, add 2·(8/3)·ρ·a³ manually.
            d.Mms = d.Mmd
        if d.Cms is None and d.Vas is not None:
            d.Cms = d.Vas / (RHO * C_SOUND ** 2 * d.Sd ** 2)
        if d.Cms is None and d.Fs is not None:
            d.Cms = 1.0 / ((2 * math.pi * d.Fs) ** 2 * d.Mms)
        if d.Cms is None:
            raise ValueError("Driver needs Cms, Vas, or Fs")
        if d.Fs is None:
            d.Fs = 1.0 / (2 * math.pi * math.sqrt(d.Mms * d.Cms))
        if d.Rms is None and d.Qms is not None:
            d.Rms = (2 * math.pi * d.Fs * d.Mms) / d.Qms
        if d.Rms is None:
            d.Rms = (2 * math.pi * d.Fs * d.Mms) / 5.0
        return d


# ---------------------------------------------------------------------------
# Port and chamber
# ---------------------------------------------------------------------------

@dataclass
class Port:
    """Vented port (acoustic mass + radiation loading + small viscous loss).

    `area` is the combined area of all parallel openings and `length` is
    the physical port length. End corrections are added per physical
    opening via `flanged_inside`/`flanged_outside`.

    `n_parallel` handles split-but-identical ports. The equivalent acoustic
    mass uses the total area, but the end correction is computed from the
    per-port area. This keeps old single-port callers unchanged while
    avoiding the combined-area end-correction error for split topologies.

    For an "open" front chamber where the chamber face is essentially open
    to outside (Hornresp Lp ≈ 0 with large area), use length=0 with both
    flanges True — only end corrections contribute inertance.
    """
    area: float
    length: float
    flanged_inside: bool = True
    flanged_outside: bool = True
    Q_port: float = 50.0
    n_parallel: int = 1

    def end_correction(self, flanged: bool) -> float:
        # Delegate to the canonical primitive in ``port_acoustics`` so the
        # 0.85 / 0.61 Rayleigh constants live in exactly one place.
        return _pa_end_correction(
            self.area,
            "flanged" if flanged else "free",
            n_parallel=self.n_parallel,
        )

    @property
    def L_eff(self) -> float:
        return (self.length
                + self.end_correction(self.flanged_inside)
                + self.end_correction(self.flanged_outside))

    @property
    def Mport(self) -> float:
        return RHO * self.L_eff / self.area

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z_m = 1j * omega * self.Mport
        R_visc = omega * self.Mport / self.Q_port
        # Radiation resistance into half-space (real part of piston Z).
        # End correction supplies the reactive (mass) part — don't double
        # count by adding piston-impedance reactance here.
        if self.flanged_outside:
            R_rad = RHO * omega ** 2 / (2 * math.pi * C_SOUND)
        else:
            R_rad = RHO * omega ** 2 / (4 * math.pi * C_SOUND)
        R_rad = R_rad / self.n_parallel
        return Z_m + R_visc + R_rad


@dataclass
class Chamber:
    """Sealed or ported air volume."""
    volume: float
    fill_loss: float = 0.0   # extra acoustic resistance Pa·s/m^3 (stuffing)
    port: Optional[Port] = None

    @property
    def Cab(self) -> float:
        return self.volume / (RHO * C_SOUND ** 2)

    def load_impedance(self, omega: np.ndarray) -> np.ndarray:
        s = 1j * omega
        Y = s * self.Cab
        if self.fill_loss > 0:
            Y = Y + 1.0 / self.fill_loss
        if self.port is not None:
            Y = Y + 1.0 / self.port.impedance(omega)
        return 1.0 / Y


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class Sim:
    freq: np.ndarray
    spl_total: np.ndarray
    spl_front_port: np.ndarray
    spl_rear_port: Optional[np.ndarray]
    spl_driver_direct: np.ndarray
    phase_total: np.ndarray
    Z_electrical: np.ndarray
    U_driver: np.ndarray
    U_front_port: np.ndarray
    U_rear_port: Optional[np.ndarray]
    omega: np.ndarray
    cone_excursion_mm: np.ndarray


def simulate(
    driver: Driver,
    front_chamber: Chamber,
    rear_chamber: Chamber,
    freq: np.ndarray,
    Rg: float = 0.0,
    v_g: float = 2.83,
    radiation_half_space: bool = True,
    distance_m: float = 1.0,
    sum_ports_coherently: bool = True,
    driver_radiates_directly: bool = False,
) -> Sim:
    """Solve the lumped network and return on-axis pressure response.

    Sign convention:
      U_d > 0 means cone moves "forward" — pushes volume into the front
      chamber and pulls volume from the rear chamber.

    With Z_load_f = front-chamber-as-seen-from-driver (parallel of front
    compliance and front port), and Z_load_r similarly for rear, the
    driver equation in acoustic domain is

        p_g = (Z_drv + Z_load_f + Z_load_r) · U_d

    Front chamber pressure  p_f =  Z_load_f · U_d
    Rear  chamber pressure  p_r = -Z_load_r · U_d   (rear is depressurized)

    Port volume velocities (positive = outward to free air):
        U_pf = p_f / Z_port_f
        U_pr = p_r / Z_port_r
    """
    driver = driver.derive()
    n = driver.n_drivers
    omega = 2 * math.pi * freq
    s = 1j * omega

    # n parallel-wired identical drivers, treated as one equivalent
    # super-driver referred to combined Sd_eff = n·Sd.
    #
    # Mass and compliance (mechanical) for n cones moving in unison:
    #   Mms_total = n·Mms    (n masses), Cms_total = Cms/n (springs in parallel),
    #   Rms_total = n·Rms,   Sd_eff = n·Sd.
    # Acoustic-domain values referred to Sd_eff:
    #   Mas_eff = Mms_total/Sd_eff² = Mms/(n·Sd²)
    #   Cas_eff = Cms_total·Sd_eff² = n·Cms·Sd²
    #   Ras_eff = Rms_total/Sd_eff² = Rms/(n·Sd²)
    # Electrical (parallel-wired): Re_eff = Re/n, Le_eff = Le/n; Bl is
    # per-driver but total force on Sd_eff is F_total = Bl·i_amp, where
    # i_amp = (v_g - Bl·v)/Re_eff. The Thevenin acoustic source then
    # collapses to the same shape as a single driver with Sd→Sd_eff and
    # Z_e→Z_e_eff:
    #   p_g  = Bl·v_g / (Sd_eff·Z_e_eff)
    #   Z_em = Bl²    / (Sd_eff²·Z_e_eff)
    Sd_eff = driver.Sd * n
    Mas_eff = driver.Mms / (n * driver.Sd ** 2)
    Cas_eff = n * driver.Cms * driver.Sd ** 2
    Ras_eff = driver.Rms / (n * driver.Sd ** 2)
    Re_eff = driver.Re / n
    Le_eff = driver.Le / n
    Bl = driver.Bl

    Z_e = Re_eff + s * Le_eff + Rg
    Z_em = (Bl ** 2) / (Sd_eff ** 2 * Z_e)
    Z_drv = Ras_eff + s * Mas_eff + 1.0 / (s * Cas_eff) + Z_em
    p_g = (Bl * v_g) / (Sd_eff * Z_e)

    Z_load_f = front_chamber.load_impedance(omega)
    Z_load_r = rear_chamber.load_impedance(omega)

    Ud = p_g / (Z_drv + Z_load_f + Z_load_r)
    p_f = Z_load_f * Ud
    p_r = -Z_load_r * Ud

    if front_chamber.port is not None:
        U_pf = p_f / front_chamber.port.impedance(omega)
    else:
        U_pf = np.zeros_like(omega, dtype=complex)
    if rear_chamber.port is not None:
        U_pr = p_r / rear_chamber.port.impedance(omega)
    else:
        U_pr = None

    omega_rad = 2 * math.pi if radiation_half_space else 4 * math.pi

    def _spl_complex(U: np.ndarray) -> np.ndarray:
        # |p| = ω·ρ·|U| / (Ω·r) for a small monopole at distance r.
        return omega * RHO * U / (omega_rad * distance_m) * 1j

    p_pf = _spl_complex(U_pf)
    p_pr = _spl_complex(U_pr) if U_pr is not None else None
    p_drv_direct = _spl_complex(Ud) if driver_radiates_directly else None

    def _spl_db(p_complex: np.ndarray) -> np.ndarray:
        return 20 * np.log10(np.maximum(np.abs(p_complex), 1e-30) / P_REF)

    spl_pf = _spl_db(p_pf)
    spl_pr = _spl_db(p_pr) if p_pr is not None else None
    spl_drv = _spl_db(p_drv_direct) if p_drv_direct is not None \
              else np.full_like(spl_pf, -200.0)

    if sum_ports_coherently:
        p_total = p_pf.copy()
        if p_pr is not None:
            p_total = p_total + p_pr
        if p_drv_direct is not None:
            p_total = p_total + p_drv_direct
    else:
        # power sum (incoherent) — diagnostic only
        mag2 = np.abs(p_pf) ** 2
        if p_pr is not None:
            mag2 = mag2 + np.abs(p_pr) ** 2
        if p_drv_direct is not None:
            mag2 = mag2 + np.abs(p_drv_direct) ** 2
        p_total = np.sqrt(mag2).astype(complex)

    spl_total = _spl_db(p_total)
    phase_total = np.degrees(np.angle(p_total))

    # Electrical impedance: V/I including back-EMF.
    v_cone = Ud / Sd_eff
    I = (v_g - Bl * v_cone) / (Re_eff + s * Le_eff + Rg)
    Z_elec = v_g / I

    # Cone excursion at v_g drive: |x| = |U_d| / (Sd_eff · ω)
    x_peak_mm = np.abs(Ud) / (Sd_eff * omega) * 1000.0

    return Sim(
        freq=freq,
        spl_total=spl_total,
        spl_front_port=spl_pf,
        spl_rear_port=spl_pr,
        spl_driver_direct=spl_drv,
        phase_total=phase_total,
        Z_electrical=Z_elec,
        U_driver=Ud,
        U_front_port=U_pf,
        U_rear_port=U_pr,
        omega=omega,
        cone_excursion_mm=x_peak_mm,
    )


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def bp4_sealed_rear(
    driver: Driver,
    Vb_front: float,
    front_port: Port,
    Vb_rear: float,
    freq: np.ndarray,
    **kwargs,
) -> Sim:
    """4th-order bandpass: sealed rear chamber + ported front chamber.

    Output exits via the front port only.
    """
    return simulate(
        driver=driver,
        front_chamber=Chamber(volume=Vb_front, port=front_port),
        rear_chamber=Chamber(volume=Vb_rear, port=None),
        freq=freq,
        **kwargs,
    )


def bp6s_dual_ported(
    driver: Driver,
    Vb_front: float,
    front_port: Port,
    Vb_rear: float,
    rear_port: Port,
    freq: np.ndarray,
    **kwargs,
) -> Sim:
    """6th-order bandpass series: ported front + ported rear.

    Both ports radiate to outside, summed coherently as if coincident.
    For spatially separated ports, post-process the per-port complex
    pressures with appropriate path-length phase from each port to the
    listening point.
    """
    return simulate(
        driver=driver,
        front_chamber=Chamber(volume=Vb_front, port=front_port),
        rear_chamber=Chamber(volume=Vb_rear, port=rear_port),
        freq=freq,
        **kwargs,
    )


def log_freq(f_min: float = 10.0, f_max: float = 2000.0,
             n: int = 1000) -> np.ndarray:
    return np.logspace(math.log10(f_min), math.log10(f_max), n)
