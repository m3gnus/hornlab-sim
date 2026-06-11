"""Degenerate-case parity test for the native Metal LEM->BEM backend."""

from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_single_aperture_unit_velocity_matches_canonical_metal():
    """U = A * v_unit -> coupled run == direct Metal BEM with v_n = v_unit."""
    pytest.importorskip("gmsh")
    metal = pytest.importorskip("hornlab_metal_bem")

    try:
        from hornlab_metal_bem.metal import discover_native_runtime

        runtime = discover_native_runtime(run_smoke_test=True)
    except Exception as exc:
        pytest.skip(f"hornlab_metal_bem native runtime unavailable: {exc}")
    if not runtime.available:
        pytest.skip("hornlab_metal_bem native runtime unavailable")

    from hornlab_metal_bem.config import VelocityMode
    from hornlab_metal_bem.mesh import load_mesh
    from hornlab_sim.methods import lem_to_bem
    from hornlab_sim.methods.lem_to_bem import _aperture_face_areas

    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "sphere.msh"
        _build_sphere(mesh_path, radius_m=0.05, elem_size_m=0.025)

        config = metal.native_config(
            formulation="complex_k",
            freq_count=1,
            freq_min_hz=500.0,
            freq_max_hz=500.0,
        )

        freqs = np.array([500.0, 1000.0])
        loaded = load_mesh(mesh_path)
        area_m2 = _aperture_face_areas(loaded, {"src": [2]})["src"]

        v_unit = 0.123 + 0.0j
        U = np.full(len(freqs), area_m2 * v_unit, dtype=np.complex128)

        result_coupled = lem_to_bem.solve(
            mesh=loaded,
            lem_velocities={"src": U},
            aperture_tags={"src": [2]},
            frequencies_hz=freqs,
            config=config,
        )

        direct_config = replace(
            config,
            velocity_mode=VelocityMode.VELOCITY,
            velocity_sources={2: complex(v_unit)},
        )
        result_direct = metal.solve_frequencies(loaded, freqs, direct_config)

        np.testing.assert_allclose(
            result_coupled.pressure_complex,
            result_direct.pressure_complex,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Coupled lem_to_bem pressure differs from direct Metal BEM",
        )
        np.testing.assert_allclose(
            result_coupled.impedance,
            result_direct.impedance,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Coupled lem_to_bem impedance differs from direct Metal BEM",
        )


def _build_sphere(path: Path, radius_m: float, elem_size_m: float) -> None:
    """Closed sphere mesh, single surface physical group tag 2."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("metal_parity_sphere")
        vol = gmsh.model.occ.addSphere(0, 0, 0, radius_m)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getBoundary([(3, vol)], oriented=False)
        gmsh.model.occ.remove([(3, vol)], recursive=False)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)
        gmsh.model.setPhysicalName(2, 2, "source")
        gmsh.option.setNumber("Mesh.MeshSizeMax", elem_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", elem_size_m)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
