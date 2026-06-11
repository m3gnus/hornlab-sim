"""Lazy BEM backend dispatch for hornlab-sim coupling methods."""

from __future__ import annotations

import os
from dataclasses import fields
from types import SimpleNamespace
from typing import Any


BackendName = str


def resolve_backend(config: Any | None = None) -> BackendName:
    """Return the BEM backend name for a config or the current environment."""
    if config is not None:
        module = type(config).__module__
        if module.startswith("hornlab_metal_bem"):
            return "metal"
        if module.startswith("hornlab_bempp_bem"):
            return "bempp"
        raise ValueError(
            f"Cannot infer BEM backend from config type {type(config)!r}; "
            "expected hornlab_metal_bem or hornlab_bempp_bem config"
        )

    requested = os.environ.get("HORNLAB_SIM_BEM_BACKEND", "auto").strip().lower()
    if requested in {"metal", "bempp"}:
        return requested
    if requested not in {"", "auto"}:
        raise ValueError(
            "HORNLAB_SIM_BEM_BACKEND must be 'auto', 'metal', or 'bempp', "
            f"got {requested!r}"
        )

    try:
        from hornlab_metal_bem.metal import discover_native_runtime

        if discover_native_runtime(run_smoke_test=True).available:
            return "metal"
    except Exception:
        pass
    return "bempp"


def backend_api(backend: BackendName):
    """Lazy-import and return the API namespace for ``backend``."""
    if backend == "metal":
        import hornlab_metal_bem as metal
        from hornlab_metal_bem.config import VelocityMode

        def default_config(formulation: str | None):
            if formulation is None:
                return metal.native_config()
            return metal.native_config(formulation=formulation)

        return SimpleNamespace(
            name="metal",
            load_mesh=metal.load_mesh,
            solve_frequencies=metal.solve_frequencies,
            VelocityMode=VelocityMode,
            default_config=default_config,
        )

    if backend == "bempp":
        import hornlab_bempp_bem as bempp
        from hornlab_bempp_bem.config import BIEFormulation, VelocityMode

        def default_config(formulation: str | None):
            if formulation is None:
                return bempp.SolveConfig()
            if formulation == "complex_k":
                return bempp.SolveConfig(formulation=BIEFormulation.COMPLEX_K)
            if formulation == "standard":
                return bempp.SolveConfig(formulation=BIEFormulation.STANDARD)
            return bempp.SolveConfig(formulation=formulation)

        return SimpleNamespace(
            name="bempp",
            load_mesh=bempp.load_mesh,
            solve_frequencies=bempp.solve_frequencies,
            VelocityMode=VelocityMode,
            default_config=default_config,
        )

    raise ValueError(f"Unknown BEM backend {backend!r}; expected 'metal' or 'bempp'")


def normalized_spl_field_name(result: Any) -> str:
    """Return the dataclass field name used for normalized directivity."""
    field_names = {field.name for field in fields(type(result))}
    return "directivity_db" if "directivity_db" in field_names else "spl_db"
