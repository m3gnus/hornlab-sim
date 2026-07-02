"""Voltage-driven driver coupling for BEM-terminated cardioid branches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import bandpass, radiation_impedance


@dataclass(frozen=True)
class CoupledCardioidResult:
    frequencies_hz: NDArray[np.float64]
    cone_volume_velocity: NDArray[np.complex128]
    port_volume_velocity: NDArray[np.complex128]
    port_to_cone_ratio: NDArray[np.complex128]
    acoustic_load: NDArray[np.complex128]
    electrical_input_impedance: NDArray[np.complex128]
    cone_excursion_m: NDArray[np.float64]
    mmd_correction_kg: float
    diagnostics: dict


@dataclass(frozen=True)
class CoupledDirectRadiatorResult:
    frequencies_hz: NDArray[np.float64]
    cone_volume_velocity: NDArray[np.complex128]
    acoustic_load: NDArray[np.complex128]
    electrical_input_impedance: NDArray[np.complex128]
    cone_excursion_m: NDArray[np.float64]
    mmd_correction_kg: float
    diagnostics: dict


def coupled_cardioid_response(
    frequencies_hz,
    *,
    driver: bandpass.Driver,
    z_mm,
    z_mf_from_port=None,
    z_port_from_mf=None,
    termination_load,
    chamber_volume_m3,
    port_area_m2,
    port_length_m,
    interior_end_correction_length_m=0.0,
    series_resistance_pa_s_m3=0.0,
    rear_sign=1.0,
    drive_voltage_v=2.83,
    rg_ohm=0.0,
    rho=radiation_impedance.RHO_AIR,
    c=radiation_impedance.C_AIR,
) -> CoupledCardioidResult:
    """Solve cone and port velocities for a BEM-terminated cardioid branch.

    Inputs use engineering-convention acoustic impedances with
    ``e^{+j omega t}`` phasors.  ``drive_voltage_v`` is an RMS generator
    voltage.  The branch ratio is delegated to
    ``radiation_impedance.terminated_chamber_port_branch`` so this helper uses
    the same chamber/port algebra as the fixed-velocity path.
    """
    freqs = _validate_frequencies(frequencies_hz)
    z_self = _validate_complex_array("z_mm", z_mm, freqs)
    z_mf_port = _validate_optional_complex_array(
        "z_mf_from_port",
        z_mf_from_port,
        freqs,
    )
    z_port_mf = _validate_optional_complex_array(
        "z_port_from_mf",
        z_port_from_mf,
        freqs,
    )
    volume = _positive_finite("chamber_volume_m3", chamber_volume_m3)
    rho_f = _positive_finite("rho", rho)
    c_f = _positive_finite("c", c)
    rear = _validate_rear_sign(rear_sign)
    drive_voltage = _finite_float("drive_voltage_v", drive_voltage_v)
    rg = _nonnegative_finite("rg_ohm", rg_ohm)

    branch = radiation_impedance.terminated_chamber_port_branch(
        freqs,
        termination_load,
        chamber_volume_m3=volume,
        port_area_m2=port_area_m2,
        port_length_m=port_length_m,
        interior_end_correction_length_m=interior_end_correction_length_m,
        series_resistance_pa_s_m3=series_resistance_pa_s_m3,
        rho=rho_f,
        c=c_f,
    )

    derived = driver.derive(rho=rho_f, c=c_f)
    n = _positive_int("driver.n_drivers", derived.n_drivers)
    sd = _positive_finite("driver.Sd", derived.Sd)
    bl = _positive_finite("driver.Bl", derived.Bl)
    cms = _positive_finite("driver.Cms", derived.Cms)
    rms = _nonnegative_finite("driver.Rms", derived.Rms)
    mmd_eff, mmd_correction, mmd_source = _effective_mmd(
        driver,
        derived,
        rho_f,
    )

    omega = 2.0 * np.pi * freqs
    s = 1j * omega
    compliance = volume / (rho_f * c_f * c_f)
    ratio = branch.exit_to_input_volume_velocity_ratio
    exterior_drive = -1j * omega * compliance * z_port_mf
    port_to_cone = ratio * (rear + exterior_drive)
    chamber_load = rear * (rear - port_to_cone) / (s * compliance)
    acoustic_load = z_self + z_mf_port * port_to_cone + chamber_load

    sd_eff = n * sd
    mas_eff = mmd_eff / (n * sd ** 2)
    cas_eff = n * cms * sd ** 2
    ras_eff = rms / (n * sd ** 2)

    d_eff = bandpass.Driver(
        Sd=sd,
        Bl=bl,
        Re=derived.Re / n,
        Le=derived.Le / n,
        le2_h=(None if derived.le2_h is None else derived.le2_h / n),
        re2_ohm=(None if derived.re2_ohm is None else derived.re2_ohm / n),
        Mms=mmd_eff,
        Cms=cms,
        Rms=rms,
    )
    z_e = d_eff.blocked_electrical_impedance(omega, Rg=rg)
    z_e_blocked = d_eff.blocked_electrical_impedance(omega, Rg=0.0)
    z_em = bl ** 2 / (sd_eff ** 2 * z_e)
    z_drv = ras_eff + s * mas_eff + 1.0 / (s * cas_eff) + z_em
    p_g = bl * drive_voltage / (sd_eff * z_e)
    z_total = z_drv + acoustic_load
    cone_velocity = p_g / z_total
    port_velocity = port_to_cone * cone_velocity
    electrical_input = z_e_blocked + bl ** 2 / (
        sd_eff ** 2 * (z_total - z_em)
    )
    cone_excursion = np.abs(cone_velocity) / (omega * sd_eff)

    diagnostics: dict[str, Any] = {
        "branch_input_impedance": branch.input_impedance,
        "branch_exit_to_input_volume_velocity_ratio": ratio,
        "blocked_electrical_impedance_with_rg": z_e,
        "blocked_electrical_impedance_ohm": z_e_blocked,
        "electromechanical_impedance": z_em,
        "driver_acoustic_impedance": z_drv,
        "chamber_compliance": compliance,
        "sd_eff_m2": sd_eff,
        "mas_eff": mas_eff,
        "cas_eff": cas_eff,
        "ras_eff": ras_eff,
        "mmd_eff_kg": mmd_eff,
        "mmd_source": mmd_source,
        "mmd_correction_fraction": (
            0.0 if mmd_correction == 0.0 else mmd_correction / derived.Mms
        ),
        "rear_sign": rear,
    }

    return CoupledCardioidResult(
        frequencies_hz=np.array(freqs, dtype=np.float64, copy=True),
        cone_volume_velocity=np.asarray(cone_velocity, dtype=np.complex128),
        port_volume_velocity=np.asarray(port_velocity, dtype=np.complex128),
        port_to_cone_ratio=np.asarray(port_to_cone, dtype=np.complex128),
        acoustic_load=np.asarray(acoustic_load, dtype=np.complex128),
        electrical_input_impedance=np.asarray(
            electrical_input,
            dtype=np.complex128,
        ),
        cone_excursion_m=np.asarray(cone_excursion, dtype=np.float64),
        mmd_correction_kg=float(mmd_correction),
        diagnostics=diagnostics,
    )


def coupled_direct_radiator_response(
    frequencies_hz,
    *,
    driver: bandpass.Driver,
    z_self,
    rear_chamber_volume_m3=None,
    drive_voltage_v=2.83,
    rg_ohm=0.0,
    rho=radiation_impedance.RHO_AIR,
    c=radiation_impedance.C_AIR,
) -> CoupledDirectRadiatorResult:
    """Solve a voltage-driven direct radiator against a BEM self-load.

    Inputs use engineering-convention acoustic impedances with
    ``e^{+j omega t}`` phasors.  ``z_self`` is the driver's acoustic
    radiation self-impedance.  When ``rear_chamber_volume_m3`` is supplied,
    a sealed-box compliance is added in series with that external load.
    """
    freqs = _validate_frequencies(frequencies_hz)
    acoustic_load = _validate_complex_array("z_self", z_self, freqs)
    rho_f = _positive_finite("rho", rho)
    c_f = _positive_finite("c", c)
    drive_voltage = _finite_float("drive_voltage_v", drive_voltage_v)
    rg = _nonnegative_finite("rg_ohm", rg_ohm)

    omega = 2.0 * np.pi * freqs
    s = 1j * omega
    if rear_chamber_volume_m3 is None:
        rear_compliance = None
    else:
        rear_volume = _positive_finite(
            "rear_chamber_volume_m3",
            rear_chamber_volume_m3,
        )
        rear_compliance = rear_volume / (rho_f * c_f * c_f)
        acoustic_load = acoustic_load + 1.0 / (s * rear_compliance)

    derived = driver.derive(rho=rho_f, c=c_f)
    n = _positive_int("driver.n_drivers", derived.n_drivers)
    sd = _positive_finite("driver.Sd", derived.Sd)
    bl = _positive_finite("driver.Bl", derived.Bl)
    cms = _positive_finite("driver.Cms", derived.Cms)
    rms = _nonnegative_finite("driver.Rms", derived.Rms)
    mmd_eff, mmd_correction, mmd_source = _effective_mmd(
        driver,
        derived,
        rho_f,
    )

    sd_eff = n * sd
    mas_eff = mmd_eff / (n * sd ** 2)
    cas_eff = n * cms * sd ** 2
    ras_eff = rms / (n * sd ** 2)

    d_eff = bandpass.Driver(
        Sd=sd,
        Bl=bl,
        Re=derived.Re / n,
        Le=derived.Le / n,
        le2_h=(None if derived.le2_h is None else derived.le2_h / n),
        re2_ohm=(None if derived.re2_ohm is None else derived.re2_ohm / n),
        Mms=mmd_eff,
        Cms=cms,
        Rms=rms,
    )
    z_e = d_eff.blocked_electrical_impedance(omega, Rg=rg)
    z_e_blocked = d_eff.blocked_electrical_impedance(omega, Rg=0.0)
    z_em = bl ** 2 / (sd_eff ** 2 * z_e)
    z_drv = ras_eff + s * mas_eff + 1.0 / (s * cas_eff) + z_em
    p_g = bl * drive_voltage / (sd_eff * z_e)
    z_total = z_drv + acoustic_load
    cone_velocity = p_g / z_total
    electrical_input = z_e_blocked + bl ** 2 / (
        sd_eff ** 2 * (z_total - z_em)
    )
    cone_excursion = np.abs(cone_velocity) / (omega * sd_eff)

    diagnostics: dict[str, Any] = {
        "blocked_electrical_impedance_with_rg": z_e,
        "blocked_electrical_impedance_ohm": z_e_blocked,
        "electromechanical_impedance": z_em,
        "driver_acoustic_impedance": z_drv,
        "rear_chamber_compliance": rear_compliance,
        "sd_eff_m2": sd_eff,
        "mas_eff": mas_eff,
        "cas_eff": cas_eff,
        "ras_eff": ras_eff,
        "mmd_eff_kg": mmd_eff,
        "mmd_source": mmd_source,
        "mmd_correction_fraction": (
            0.0 if mmd_correction == 0.0 else mmd_correction / derived.Mms
        ),
    }

    return CoupledDirectRadiatorResult(
        frequencies_hz=np.array(freqs, dtype=np.float64, copy=True),
        cone_volume_velocity=np.asarray(cone_velocity, dtype=np.complex128),
        acoustic_load=np.asarray(acoustic_load, dtype=np.complex128),
        electrical_input_impedance=np.asarray(
            electrical_input,
            dtype=np.complex128,
        ),
        cone_excursion_m=np.asarray(cone_excursion, dtype=np.float64),
        mmd_correction_kg=float(mmd_correction),
        diagnostics=diagnostics,
    )


def _effective_mmd(
    driver: bandpass.Driver,
    derived: bandpass.Driver,
    rho: float,
) -> tuple[float, float, str]:
    if driver.Mmd is not None:
        return _positive_finite("driver.Mmd", driver.Mmd), 0.0, "Mmd"

    mms = _positive_finite("driver.Mms", derived.Mms)
    sd = _positive_finite("driver.Sd", derived.Sd)
    radius = math.sqrt(sd / math.pi)
    correction = 2.0 * (8.0 / 3.0) * rho * radius ** 3
    if correction > 0.30 * mms:
        raise ValueError(
            "Mms free-air radiation-mass correction exceeds 30% of Mms "
            f"({correction / mms:.3f}); check Sd/Mms"
        )
    mmd = mms - correction
    if not math.isfinite(mmd) or mmd <= 0.0:
        raise ValueError(
            f"Mmd after free-air radiation-mass correction must be positive, got {mmd!r}"
        )
    return mmd, correction, "Mms_corrected"


def _validate_frequencies(values) -> NDArray[np.float64]:
    freqs = np.asarray(values, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies_hz is empty")
    bad = freqs[~np.isfinite(freqs) | (freqs <= 0.0)]
    if bad.size:
        raise ValueError(f"frequencies_hz must be positive finite values: {bad[:5]}")
    return freqs


def _validate_complex_array(
    name: str,
    values,
    freqs: NDArray[np.float64],
) -> NDArray[np.complex128]:
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size == 1 and freqs.size != 1:
        arr = np.full(freqs.shape, arr[0], dtype=np.complex128)
    if arr.shape != freqs.shape:
        raise ValueError(f"{name} shape {arr.shape}, expected {freqs.shape}")
    if not np.all(np.isfinite(arr.real) & np.isfinite(arr.imag)):
        raise ValueError(f"{name} must contain finite complex values")
    return arr


def _validate_optional_complex_array(
    name: str,
    values,
    freqs: NDArray[np.float64],
) -> NDArray[np.complex128]:
    if values is None:
        return np.zeros(freqs.shape, dtype=np.complex128)
    return _validate_complex_array(name, values, freqs)


def _positive_finite(name: str, value) -> float:
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value_f


def _nonnegative_finite(name: str, value) -> float:
    value_f = float(value)
    if not math.isfinite(value_f) or value_f < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return value_f


def _finite_float(name: str, value) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value_f


def _positive_int(name: str, value) -> int:
    value_i = int(value)
    if value_i != value or value_i <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value_i


def _validate_rear_sign(value) -> float:
    sign = float(value)
    if sign not in (-1.0, 1.0):
        raise ValueError(f"rear_sign must be +1.0 or -1.0, got {value!r}")
    return sign
