from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_sim.methods import driver_coupling, radiation_impedance
from hornlab_sim.methods.bandpass import Driver


def _base_driver(**overrides):
    params = dict(
        Sd=0.013,
        Bl=5.8,
        Re=5.6,
        Le=0.25e-3,
        Mmd=0.018,
        Cms=0.85e-3,
        Rms=1.2,
    )
    params.update(overrides)
    return Driver(**params)


def _common_kwargs(freqs, **overrides):
    kwargs = dict(
        frequencies_hz=freqs,
        driver=_base_driver(),
        z_mm=24.0 + 1j * 0.02 * (2.0 * np.pi * freqs),
        termination_load=15.0 + 1j * 0.01 * (2.0 * np.pi * freqs),
        chamber_volume_m3=6.0e-3,
        port_area_m2=7.5e-4,
        port_length_m=0.035,
        interior_end_correction_length_m=0.002,
        series_resistance_pa_s_m3=8.0,
    )
    kwargs.update(overrides)
    return kwargs


def _hand_driver_terms(
    driver: Driver,
    freqs,
    *,
    drive_voltage_v=2.83,
    rg_ohm=0.0,
    rho=radiation_impedance.RHO_AIR,
    c=radiation_impedance.C_AIR,
):
    d = driver.derive(rho=rho, c=c)
    n = d.n_drivers
    omega = 2.0 * np.pi * freqs
    s = 1j * omega
    sd_eff = n * d.Sd
    mas_eff = d.Mms / (n * d.Sd ** 2)
    cas_eff = n * d.Cms * d.Sd ** 2
    ras_eff = d.Rms / (n * d.Sd ** 2)
    d_eff = Driver(
        Sd=d.Sd,
        Bl=d.Bl,
        Re=d.Re / n,
        Le=d.Le / n,
        le2_h=(None if d.le2_h is None else d.le2_h / n),
        re2_ohm=(None if d.re2_ohm is None else d.re2_ohm / n),
        Mms=d.Mms,
        Cms=d.Cms,
        Rms=d.Rms,
    )
    z_e = d_eff.blocked_electrical_impedance(omega, Rg=rg_ohm)
    z_em = d.Bl ** 2 / (sd_eff ** 2 * z_e)
    z_drv = ras_eff + s * mas_eff + 1.0 / (s * cas_eff) + z_em
    p_g = d.Bl * drive_voltage_v / (sd_eff * z_e)
    return {
        "omega": omega,
        "sd_eff": sd_eff,
        "mas_eff": mas_eff,
        "cas_eff": cas_eff,
        "z_drv": z_drv,
        "z_e": z_e,
        "p_g": p_g,
    }


def _direct_peak_frequency(freqs, *, driver: Driver, **kwargs):
    result = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=np.zeros(freqs.shape, dtype=np.complex128),
        **kwargs,
    )
    peak_freq = freqs[
        int(np.argmax(np.abs(result.electrical_input_impedance)))
    ]
    return peak_freq, result


def test_fixed_velocity_ratio_invariant_for_both_rear_polarities():
    freqs = np.array([160.0, 315.0, 630.0, 1000.0])
    z_port_from_mf = np.array([4.0 + 2.0j, 5.0 + 3.0j, 6.0 + 4.0j, 7.0 + 5.0j])
    kwargs = _common_kwargs(
        freqs,
        z_port_from_mf=z_port_from_mf,
        z_mf_from_port=np.array([3.0 + 1.0j, 4.0 + 2.0j, 5.0 + 3.0j, 6.0 + 4.0j]),
    )

    branch = radiation_impedance.terminated_chamber_port_branch(
        freqs,
        kwargs["termination_load"],
        chamber_volume_m3=kwargs["chamber_volume_m3"],
        port_area_m2=kwargs["port_area_m2"],
        port_length_m=kwargs["port_length_m"],
        interior_end_correction_length_m=kwargs["interior_end_correction_length_m"],
        series_resistance_pa_s_m3=kwargs["series_resistance_pa_s_m3"],
    )
    omega = 2.0 * np.pi * freqs
    compliance = kwargs["chamber_volume_m3"] / (
        radiation_impedance.RHO_AIR * radiation_impedance.C_AIR ** 2
    )

    for rear_sign in (1.0, -1.0):
        result = driver_coupling.coupled_cardioid_response(
            **kwargs,
            rear_sign=rear_sign,
        )
        expected = branch.exit_to_input_volume_velocity_ratio * (
            rear_sign - 1j * omega * compliance * z_port_from_mf
        )
        np.testing.assert_allclose(
            result.port_to_cone_ratio,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_sealed_box_limit_matches_hand_composition_and_resonance():
    driver = _base_driver(Bl=1.2, Le=0.0, Mmd=0.020, Cms=1.0e-3, Rms=0.02)
    freqs = np.linspace(40.0, 140.0, 2001)
    volume = 5.0e-3
    z_mm = np.zeros(freqs.shape, dtype=np.complex128)
    result = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(
            freqs,
            driver=driver,
            z_mm=z_mm,
            termination_load=np.zeros(freqs.shape, dtype=np.complex128),
            chamber_volume_m3=volume,
            series_resistance_pa_s_m3=1.0e14,
        )
    )

    terms = _hand_driver_terms(driver, freqs)
    s = 1j * terms["omega"]
    c_box = volume / (radiation_impedance.RHO_AIR * radiation_impedance.C_AIR ** 2)
    expected_ud = terms["p_g"] / (terms["z_drv"] + 1.0 / (s * c_box))
    np.testing.assert_allclose(
        result.cone_volume_velocity,
        expected_ud,
        rtol=5e-8,
        atol=1e-12,
    )

    zmag = np.abs(result.electrical_input_impedance)
    peak_freq = freqs[int(np.argmax(zmag))]
    cas_par = terms["cas_eff"] * c_box / (terms["cas_eff"] + c_box)
    fc = 1.0 / (2.0 * math.pi * math.sqrt(terms["mas_eff"] * cas_par))
    assert abs(peak_freq - fc) <= np.max(np.diff(freqs))


def test_passive_loads_keep_input_resistance_above_blocked_voice_coil():
    freqs = np.logspace(math.log10(80.0), math.log10(1600.0), 64)
    result = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(
            freqs,
            z_mm=20.0 + 1j * 0.015 * (2.0 * np.pi * freqs),
            termination_load=30.0 + 1j * 0.02 * (2.0 * np.pi * freqs),
            z_mf_from_port=None,
            z_port_from_mf=None,
        )
    )
    terms = _hand_driver_terms(_base_driver(), freqs)

    assert np.all(
        np.real(result.electrical_input_impedance)
        >= np.real(terms["z_e"]) - 1e-9
    )


def test_mms_and_mmd_bookkeeping_equivalence_and_guard():
    freqs = np.array([100.0, 250.0, 800.0])
    sd = 0.010
    mmd = 0.014
    radius = math.sqrt(sd / math.pi)
    correction = 2.0 * (8.0 / 3.0) * radiation_impedance.RHO_AIR * radius ** 3
    driver_mmd = _base_driver(Sd=sd, Mmd=mmd, Mms=None)
    driver_mms = _base_driver(Sd=sd, Mmd=None, Mms=mmd + correction)

    result_mmd = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, driver=driver_mmd)
    )
    result_mms = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, driver=driver_mms)
    )

    assert result_mmd.mmd_correction_kg == pytest.approx(0.0)
    assert result_mms.mmd_correction_kg == pytest.approx(correction)
    np.testing.assert_allclose(
        result_mms.cone_volume_velocity,
        result_mmd.cone_volume_velocity,
    )
    np.testing.assert_allclose(
        result_mms.electrical_input_impedance,
        result_mmd.electrical_input_impedance,
    )

    bad_driver = _base_driver(Sd=0.05, Mmd=None, Mms=0.020)
    with pytest.raises(ValueError, match="exceeds 50%"):
        driver_coupling.coupled_cardioid_response(
            **_common_kwargs(freqs, driver=bad_driver)
        )

    # A light large-cone pro woofer: the correction is ~40% of Mms, which is
    # accepted and reported through the correction fraction.
    light_driver = _base_driver(Sd=0.05, Mmd=None, Mms=0.032)
    light = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, driver=light_driver)
    )
    fraction = light.mmd_correction_kg / 0.032
    assert driver_coupling.MMD_CORRECTION_WARN < fraction < driver_coupling.MMD_CORRECTION_LIMIT


def test_lr2_blocked_impedance_and_n_driver_referred_collapse():
    freqs = np.array([120.0, 600.0, 2400.0])
    omega = 2.0 * np.pi * freqs
    driver_lr2 = _base_driver(
        le2_h=1.5e-3,
        re2_ohm=18.0,
        n_drivers=2,
    )
    result_lr2 = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, driver=driver_lr2),
        rg_ohm=0.25,
    )
    expected_ze = Driver(
        Sd=driver_lr2.Sd,
        Bl=driver_lr2.Bl,
        Re=driver_lr2.Re / 2,
        Le=driver_lr2.Le / 2,
        le2_h=driver_lr2.le2_h / 2,
        re2_ohm=driver_lr2.re2_ohm / 2,
        Mms=driver_lr2.Mmd,
        Cms=driver_lr2.Cms,
        Rms=driver_lr2.Rms,
    ).blocked_electrical_impedance(omega, Rg=0.25)
    plain = _base_driver(n_drivers=2)
    plain_ze = Driver(
        Sd=plain.Sd,
        Bl=plain.Bl,
        Re=plain.Re / 2,
        Le=plain.Le / 2,
        Mms=plain.Mmd,
        Cms=plain.Cms,
        Rms=plain.Rms,
    ).blocked_electrical_impedance(omega, Rg=0.25)

    np.testing.assert_allclose(
        result_lr2.diagnostics["blocked_electrical_impedance_with_rg"],
        expected_ze,
    )
    assert not np.allclose(expected_ze, plain_ze)

    equivalent_single = _base_driver(
        Sd=2.0 * driver_lr2.Sd,
        Re=driver_lr2.Re / 2.0,
        Le=driver_lr2.Le / 2.0,
        le2_h=driver_lr2.le2_h / 2.0,
        re2_ohm=driver_lr2.re2_ohm / 2.0,
        Mmd=2.0 * driver_lr2.Mmd,
        Cms=driver_lr2.Cms / 2.0,
        Rms=2.0 * driver_lr2.Rms,
        n_drivers=1,
    )
    result_equiv = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, driver=equivalent_single),
        rg_ohm=0.25,
    )

    np.testing.assert_allclose(
        result_lr2.cone_volume_velocity,
        result_equiv.cone_volume_velocity,
    )
    np.testing.assert_allclose(
        result_lr2.port_volume_velocity,
        result_equiv.port_volume_velocity,
    )
    np.testing.assert_allclose(
        result_lr2.electrical_input_impedance,
        result_equiv.electrical_input_impedance,
    )


def test_mutual_impedance_fallbacks_are_zero_arrays():
    freqs = np.array([90.0, 180.0, 360.0])
    zero = np.zeros(freqs.shape, dtype=np.complex128)
    result_none = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, z_mf_from_port=None, z_port_from_mf=None)
    )
    result_zero = driver_coupling.coupled_cardioid_response(
        **_common_kwargs(freqs, z_mf_from_port=zero, z_port_from_mf=zero)
    )

    np.testing.assert_allclose(
        result_none.port_to_cone_ratio,
        result_zero.port_to_cone_ratio,
    )
    np.testing.assert_allclose(
        result_none.acoustic_load,
        result_zero.acoustic_load,
    )
    np.testing.assert_allclose(
        result_none.electrical_input_impedance,
        result_zero.electrical_input_impedance,
    )


def test_direct_radiator_free_air_impedance_peak_tracks_mmd_cms_resonance():
    driver = _base_driver(Le=0.0, Rms=0.35)
    fc = 1.0 / (2.0 * math.pi * math.sqrt(driver.Mmd * driver.Cms))
    freqs = np.linspace(0.6 * fc, 1.4 * fc, 2001)

    peak_freq, result = _direct_peak_frequency(freqs, driver=driver)

    assert peak_freq == pytest.approx(fc, rel=0.02)
    assert result.mmd_correction_kg == pytest.approx(0.0)
    assert result.diagnostics["mmd_source"] == "Mmd"
    assert result.diagnostics["rear_chamber_compliance"] is None


def test_direct_radiator_rear_chamber_raises_impedance_peak_frequency():
    driver = _base_driver(Le=0.0, Rms=0.35)
    fc = 1.0 / (2.0 * math.pi * math.sqrt(driver.Mmd * driver.Cms))
    freqs = np.linspace(0.6 * fc, 2.8 * fc, 5001)

    free_peak, _ = _direct_peak_frequency(freqs, driver=driver)
    boxed_peak, boxed = _direct_peak_frequency(
        freqs,
        driver=driver,
        rear_chamber_volume_m3=5.0e-3,
    )

    assert boxed_peak > free_peak
    assert boxed.diagnostics["rear_chamber_compliance"] == pytest.approx(
        5.0e-3
        / (
            radiation_impedance.RHO_AIR
            * radiation_impedance.C_AIR
            * radiation_impedance.C_AIR
        )
    )


def test_direct_radiator_mass_controlled_velocity_and_excursion():
    driver = _base_driver(Le=0.0, Rms=0.35)
    freqs = np.array([5000.0, 10000.0])

    result = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=np.zeros(freqs.shape, dtype=np.complex128),
    )

    ratio = abs(result.cone_volume_velocity[1]) / abs(
        result.cone_volume_velocity[0]
    )
    assert ratio == pytest.approx(0.5, rel=0.05)
    omega = 2.0 * np.pi * freqs
    np.testing.assert_allclose(
        result.cone_excursion_m,
        np.abs(result.cone_volume_velocity)
        / (omega * result.diagnostics["sd_eff_m2"]),
        rtol=1e-12,
        atol=0.0,
    )


def test_direct_radiator_source_resistance_reduces_drive_only():
    driver = _base_driver(Le=0.0, Rms=0.35)
    freqs = np.array([60.0, 90.0, 200.0, 800.0, 2000.0])
    z_self = np.zeros(freqs.shape, dtype=np.complex128)

    ideal = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=z_self,
    )
    series = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=z_self,
        rg_ohm=2.0,
    )

    assert np.all(
        np.abs(series.cone_volume_velocity) < np.abs(ideal.cone_volume_velocity)
    )
    np.testing.assert_allclose(
        series.electrical_input_impedance,
        ideal.electrical_input_impedance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_direct_radiator_huge_rear_chamber_converges_to_no_rear_chamber():
    driver = _base_driver(Le=0.0, Rms=0.35)
    freqs = np.logspace(math.log10(30.0), math.log10(3000.0), 32)
    omega = 2.0 * np.pi * freqs
    z_self = 18.0 + 1j * 0.004 * omega

    open_back = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=z_self,
        rear_chamber_volume_m3=None,
    )
    huge_box = driver_coupling.coupled_direct_radiator_response(
        freqs,
        driver=driver,
        z_self=z_self,
        rear_chamber_volume_m3=1.0e12,
    )

    np.testing.assert_allclose(
        huge_box.cone_volume_velocity,
        open_back.cone_volume_velocity,
        rtol=1e-10,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        huge_box.electrical_input_impedance,
        open_back.electrical_input_impedance,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        huge_box.acoustic_load,
        open_back.acoustic_load,
        rtol=1e-10,
        atol=1e-12,
    )
