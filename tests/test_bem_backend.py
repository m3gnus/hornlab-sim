from __future__ import annotations

from dataclasses import dataclass

import pytest

from hornlab_sim.methods import _bem_backend


def test_resolve_backend_dispatches_on_metal_config_module():
    config_type = type(
        "Config",
        (),
        {"__module__": "hornlab_metal_bem.config"},
    )

    assert _bem_backend.resolve_backend(config_type()) == "metal"


def test_resolve_backend_dispatches_on_bempp_config_module():
    config_type = type(
        "Config",
        (),
        {"__module__": "hornlab_bempp_bem.config"},
    )

    assert _bem_backend.resolve_backend(config_type()) == "bempp"


def test_resolve_backend_env_bempp_forces_bempp(monkeypatch):
    monkeypatch.setenv("HORNLAB_SIM_BEM_BACKEND", "bempp")

    assert _bem_backend.resolve_backend(None) == "bempp"


def test_resolve_backend_unknown_config_type_raises():
    with pytest.raises(ValueError, match="Cannot infer BEM backend"):
        _bem_backend.resolve_backend(object())


def test_normalized_spl_field_name_prefers_directivity_dataclass_field():
    @dataclass
    class MetalLikeResult:
        directivity_db: object

    assert _bem_backend.normalized_spl_field_name(MetalLikeResult(None)) == "directivity_db"


def test_normalized_spl_field_name_falls_back_to_spl_db():
    @dataclass
    class BemppLikeResult:
        spl_db: object

        @property
        def directivity_db(self):
            return self.spl_db

    assert _bem_backend.normalized_spl_field_name(BemppLikeResult(None)) == "spl_db"
