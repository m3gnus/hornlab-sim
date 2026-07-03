"""The historical ``bigmeh_*``-prefixed provenance aliases still work.

Each alias must return exactly what its canonical replacement returns and
must emit a ``DeprecationWarning`` that names the replacement.
"""

from __future__ import annotations

import pytest

from hornlab_sim.methods import helmholtz as hh


class _SlotStub:
    """Duck-typed ``params.slot`` (see helmholtz module docstring)."""

    slot_opening_W = 130.0
    slot_depth = 500.0
    slot_height = 376.0
    apex_width_mm = 0.0

    def resolved_apex_depth(self):
        return 500.0


class _CabinetStub:
    """Duck-typed full-cabinet params object."""

    slot = _SlotStub()
    cabinet_W = 800.0
    cabinet_H = 400.0
    cabinet_D = 600.0
    slot_topology = "front"
    mids = None  # mid_chamber_helmholtz(None) falls back to defaults


_CASES = [
    ("bigmeh_slot_helmholtz", "slot_helmholtz",
     dict(opening_W_mm=130.0)),
    ("bigmeh_slot_helmholtz_from_params", "slot_helmholtz_from_params",
     dict(params=_CabinetStub())),
    ("bigmeh_mid_chamber_helmholtz", "mid_chamber_helmholtz",
     dict(chamber_volume_cc=130.0)),
    ("bigmeh_mid_chamber_helmholtz_from_params",
     "mid_chamber_helmholtz_from_params",
     dict(params=_CabinetStub())),
]


@pytest.mark.parametrize(
    "old_name, new_name, kwargs",
    _CASES,
    ids=[old for old, _, _ in _CASES],
)
def test_alias_warns_and_matches_replacement(old_name, new_name, kwargs):
    expected = getattr(hh, new_name)(**kwargs)
    with pytest.warns(
        DeprecationWarning,
        match=rf"use hornlab_sim\.methods\.helmholtz\.{new_name}\(\)",
    ):
        result = getattr(hh, old_name)(**kwargs)
    assert result == expected
