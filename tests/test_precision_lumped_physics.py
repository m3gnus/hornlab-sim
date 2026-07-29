from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_sim.methods.bandpass import Chamber, Driver, Port, simulate
from hornlab_sim.methods.helmholtz import (
    _largest_volume_for_target,
    helmholtz,
    mid_chamber_helmholtz,
)
from hornlab_sim.methods.port_acoustics import (
    confined_interior_end_correction,
    frustum_port_acoustic_mass,
    frustum_port_inertance_denominator,
    viscothermal_port_q,
)


def test_ingard_interior_correction_matches_measured_volume_shift_regression():
    """Measured clay-volume regression from an archived build thread.

    The measurement gives 4x dummy chambers changing from 130 mL to 64 mL,
    with the notch moving from 577 Hz to 713 Hz (+23.6%). The same post does
    not state every dummy-port dimension, so this pins explicit assumptions:
    one equivalent conical throat per chamber, 8.5 cm2 minimum throat
    area, 10 mm effective wall depth, and a 21.2 cm2 chamber-side flare as
    documented in the design notes (§17a) for the conical port family.
    The legacy volume-only sensitivity is independent of those dimensions.
    """
    geometry = dict(
        port_count=1,
        entry_area_cm2=8.5,
        chamber_area_cm2=21.2,
        tube_depth_mm=10.0,
        target_fc_hz=600.0,
    )
    large_legacy = mid_chamber_helmholtz(
        chamber_volume_cc=130.0,
        **geometry,
    )
    small_legacy = mid_chamber_helmholtz(
        chamber_volume_cc=64.0,
        **geometry,
    )
    large_ingard = mid_chamber_helmholtz(
        chamber_volume_cc=130.0,
        interior_end_correction="ingard",
        **geometry,
    )
    small_ingard = mid_chamber_helmholtz(
        chamber_volume_cc=64.0,
        interior_end_correction="ingard",
        **geometry,
    )

    legacy_shift = small_legacy["f_Hz"] / large_legacy["f_Hz"] - 1.0
    ingard_shift = small_ingard["f_Hz"] / large_ingard["f_Hz"] - 1.0
    measured_shift = 713.0 / 577.0 - 1.0

    assert legacy_shift == pytest.approx(math.sqrt(130.0 / 64.0) - 1.0)
    assert legacy_shift == pytest.approx(0.4252, rel=5e-4)
    assert 0.18 <= ingard_shift <= 0.30
    assert abs(ingard_shift - measured_shift) < abs(legacy_shift - measured_shift)
    assert small_ingard["interior_delta_L_m"] > large_ingard["interior_delta_L_m"]


def test_frustum_port_mass_reduces_to_uniform_tube_mass():
    rho = 1.21
    radius = 0.018
    length = 0.030
    expected = rho * length / (math.pi * radius * radius)
    assert frustum_port_acoustic_mass(radius, radius, length, rho) == pytest.approx(
        expected
    )


def test_frustum_helmholtz_frequency_rises_as_exit_flares():
    common = dict(
        chamber_volume_cc=100.0,
        port_count=1,
        entry_radius_m=0.010,
        tube_depth_mm=20.0,
        target_fc_hz=1000.0,
        port_model="frustum",
    )
    straight = mid_chamber_helmholtz(exit_radius_m=0.010, **common)
    flared = mid_chamber_helmholtz(exit_radius_m=0.020, **common)

    assert flared["f_Hz"] > straight["f_Hz"]
    assert flared["L_eff_m"] < straight["L_eff_m"]


def test_frustum_target_sizing_uses_tapered_port_inertance():
    target_hz = 1000.0
    result = mid_chamber_helmholtz(
        port_count=1,
        entry_radius_m=0.010,
        exit_radius_m=0.020,
        tube_depth_mm=20.0,
        target_fc_hz=target_hz,
        port_model="frustum",
    )

    assert result["f_Hz"] == pytest.approx(target_hz)


@pytest.mark.parametrize("port_model", ["uniform", "frustum"])
def test_ingard_target_sizing_inverts_the_selected_port_model(port_model):
    target_hz = 600.0
    result = mid_chamber_helmholtz(
        port_count=1,
        entry_area_cm2=8.5,
        chamber_area_cm2=21.2,
        tube_depth_mm=10.0,
        target_fc_hz=target_hz,
        port_model=port_model,
        interior_end_correction="ingard",
    )

    assert result["f_Hz"] == pytest.approx(target_hz, abs=1e-6)


def test_ingard_target_sizing_returns_largest_root():
    result = mid_chamber_helmholtz(
        port_count=1,
        entry_area_cm2=8.5,
        chamber_area_cm2=21.2,
        tube_depth_mm=10.0,
        target_fc_hz=728.31,
        port_model="uniform",
        interior_end_correction="ingard",
    )

    assert result["V_cc"] == pytest.approx(62.7320, rel=1e-5)
    assert result["f_Hz"] == pytest.approx(728.31, abs=1e-6)


def test_largest_root_solver_keeps_a_tangent_root():
    def frequency(volume_cc):
        return 100.0 - (volume_cc - 1.0) * (volume_cc - 2.0) ** 2

    volume_cc = _largest_volume_for_target(frequency, 100.0, 3.0)

    assert volume_cc == pytest.approx(2.0, abs=1e-6)


def test_largest_root_solver_reports_unreachable_target():
    with pytest.raises(ValueError, match="unreachable"):
        _largest_volume_for_target(lambda volume_cc: 99.0, 100.0, 3.0)


@pytest.mark.parametrize("port_model", ["uniform", "frustum"])
def test_ingard_branch_decreases_strictly_above_its_last_maximum(port_model):
    volumes_cc = np.geomspace(43.3, 200.0, 128)
    frequencies_hz = np.array(
        [
            mid_chamber_helmholtz(
                chamber_volume_cc=float(volume_cc),
                port_count=1,
                entry_area_cm2=8.5,
                chamber_area_cm2=21.2,
                tube_depth_mm=10.0,
                target_fc_hz=600.0,
                port_model=port_model,
                interior_end_correction="ingard",
            )["f_Hz"]
            for volume_cc in volumes_cc
        ]
    )
    slopes = np.diff(frequencies_hz)
    maxima = np.flatnonzero((slopes[:-1] > 0.0) & (slopes[1:] < 0.0)) + 1

    assert maxima.size == 1
    assert np.all(slopes[maxima[-1] :] < 0.0)


@pytest.mark.parametrize("port_model", ["uniform", "frustum"])
def test_ingard_without_an_interior_end_term_is_a_rayleigh_noop(port_model):
    common = dict(
        target_fc_hz=1200.0,
        end_corr="flanged",
        port_model=port_model,
    )

    rayleigh = mid_chamber_helmholtz(
        interior_end_correction="rayleigh",
        **common,
    )
    ingard = mid_chamber_helmholtz(
        interior_end_correction="ingard",
        **common,
    )

    assert ingard["V_cc"] == pytest.approx(rayleigh["V_cc"])
    assert ingard["f_Hz"] == pytest.approx(rayleigh["f_Hz"])


def test_frustum_ingard_correction_uses_the_narrow_frustum_radius():
    """The confined-neck radius is the narrower end, either way round.

    Keying the Ingard scaling to the chamber-side radius of a flared port
    saturates the ``max_confinement`` clamp and makes the correction
    volume-independent, which is what
    ``test_frustum_ingard_keeps_the_measured_volume_shift_bracket`` guards.
    """
    chamber_volume = 100.0e-6
    for entry_radius, exit_radius in ((0.010, 0.020), (0.020, 0.010)):
        _, _, interior_delta = frustum_port_inertance_denominator(
            entry_radius,
            exit_radius,
            0.020,
            interior_end_correction="ingard",
            chamber_volume_m3=chamber_volume,
        )

        expected = confined_interior_end_correction(
            min(entry_radius, exit_radius),
            chamber_volume,
        )
        assert interior_delta == pytest.approx(expected)


def test_frustum_ingard_keeps_the_measured_volume_shift_bracket():
    """The frustum arm of the measured CAFMEH bracket must stay a bracket.

    ``260611-cafmeh-measured-calibration`` records Ingard uniform (+20.96%)
    and Ingard frustum (+25.95%) straddling the measured +23.57% shift, with
    legacy Rayleigh at +42.52%. If the confined-neck correction stops
    responding to chamber volume, the frustum arm collapses back onto the
    Rayleigh value and the bracket silently disappears.
    """
    geometry = dict(
        port_count=1,
        entry_area_cm2=8.5,
        chamber_area_cm2=21.2,
        tube_depth_mm=10.0,
        target_fc_hz=600.0,
        port_model="frustum",
        interior_end_correction="ingard",
    )
    large = mid_chamber_helmholtz(chamber_volume_cc=130.0, **geometry)
    small = mid_chamber_helmholtz(chamber_volume_cc=64.0, **geometry)

    shift = small["f_Hz"] / large["f_Hz"] - 1.0
    measured_shift = 713.0 / 577.0 - 1.0
    rayleigh_shift = math.sqrt(130.0 / 64.0) - 1.0

    assert small["interior_delta_L_m"] > large["interior_delta_L_m"]
    assert shift == pytest.approx(0.2595, rel=5e-3)
    assert shift > measured_shift
    assert abs(shift - measured_shift) < 0.25 * abs(rayleigh_shift - measured_shift)


def test_frustum_ingard_narrow_radius_clamp_stays_below_anchor_volumes():
    entry_radius = math.sqrt(8.5e-4 / math.pi)
    exit_radius = math.sqrt(21.2e-4 / math.pi)
    confinement_factor = 9.0 / 8.0
    max_confinement = 0.85

    def saturation_volume_cc(radius):
        return (
            4.0
            * math.pi
            / 3.0
            * (confinement_factor * radius / max_confinement) ** 3
            * 1e6
        )

    narrow_saturation_cc = saturation_volume_cc(min(entry_radius, exit_radius))
    exit_saturation_cc = saturation_volume_cc(exit_radius)

    assert narrow_saturation_cc == pytest.approx(43.2208, rel=1e-5)
    assert narrow_saturation_cc < 64.0 < 130.0 < exit_saturation_cc


def test_geometry_derived_port_q_is_sane_and_monotonic():
    q_small = viscothermal_port_q(1000.0, 6.0e-4, hydraulic_radius_m=0.003)
    q_large = viscothermal_port_q(1000.0, 12.0e-4, hydraulic_radius_m=0.004)

    assert 10.0 <= q_small <= 40.0
    assert 10.0 <= q_large <= 40.0
    assert q_large > q_small

    port = Port(
        area=6.0e-4,
        length=0.024,
        Q_port=None,
        Q_port_eval_hz=1000.0,
    )
    assert port.derived_Q_port(1000.0) == pytest.approx(q_small)


def test_numeric_q_port_keeps_legacy_series_loss():
    omega = np.array([2.0 * math.pi * 1000.0])
    numeric = Port(area=6.0e-4, length=0.024, Q_port=50.0)
    derived = Port(
        area=6.0e-4,
        length=0.024,
        Q_port=None,
        Q_port_eval_hz=1000.0,
    )

    assert np.real(numeric.impedance(omega))[0] != pytest.approx(
        np.real(derived.impedance(omega))[0]
    )
    assert np.real(numeric.impedance(omega))[0] == pytest.approx(
        omega[0] * numeric.Mport / 50.0
        + 1.21 * omega[0] ** 2 / (2.0 * math.pi * 343.0)
    )


def test_chamber_thermal_compliance_correction_is_opt_in_and_clamped():
    omega = np.array([2.0 * math.pi * 1000.0])
    cold_omega = np.array([2.0 * math.pi * 1.0e-3])
    adiabatic = Chamber(volume=1.0e-4)
    thermal = Chamber(volume=1.0e-4, thermal_surface_area_m2=0.10)

    assert thermal.Cab == pytest.approx(adiabatic.Cab)
    assert thermal.Cab_for(omega)[0] > adiabatic.Cab
    assert thermal.Cab_for(cold_omega)[0] == pytest.approx(1.4 * adiabatic.Cab)


def test_bandpass_sweep_computes_each_port_impedance_once(monkeypatch):
    driver = Driver(
        Sd=57e-4,
        Bl=9.0,
        Re=5.5,
        Le=0.23e-3,
        Mmd=5.7e-3,
        Cms=351e-6,
        Rms=1.0,
    )
    front_port = Port(area=7.1e-4, length=0.024)
    rear_port = Port(area=4.0e-4, length=0.080)
    calls = {id(front_port): 0, id(rear_port): 0}
    original_impedance = Port.impedance

    def counted_impedance(self, omega, **kwargs):
        calls[id(self)] += 1
        return original_impedance(self, omega, **kwargs)

    monkeypatch.setattr(Port, "impedance", counted_impedance)

    simulate(
        driver,
        Chamber(volume=0.7e-3, port=front_port),
        Chamber(volume=1.2e-3, port=rear_port),
        np.logspace(np.log10(20.0), np.log10(2000.0), 64),
    )

    assert calls == {id(front_port): 1, id(rear_port): 1}


def test_lr2_blocked_electrical_impedance_matches_hand_calculation():
    driver = Driver(
        Sd=0.01,
        Bl=5.0,
        Re=6.0,
        Le=0.5e-3,
        le2_h=2.0e-3,
        re2_ohm=20.0,
        Mms=0.020,
        Cms=1.0e-3,
        Rms=1.0,
    )
    omega = np.array([2.0 * math.pi * 100.0, 2.0 * math.pi * 10000.0])
    s = 1j * omega
    expected = 6.0 + s * 0.5e-3 + (s * 2.0e-3 * 20.0) / (
        s * 2.0e-3 + 20.0
    )

    assert driver.blocked_electrical_impedance(omega) == pytest.approx(expected)
    assert np.angle(expected[1]) > np.angle(expected[0])
    assert abs(expected[1]) > abs(expected[0])


def test_air_property_override_scales_helmholtz_frequency_with_sound_speed():
    f_default = helmholtz(1.0e-3, 5.0e-4, L_geom_m=0.020)
    f_warm = helmholtz(1.0e-3, 5.0e-4, L_geom_m=0.020, c=350.0, rho=1.15)

    assert f_warm / f_default == pytest.approx(350.0 / 343.0)
