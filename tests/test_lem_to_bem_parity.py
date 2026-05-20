"""Degenerate-case parity test: lem_to_bem.solve vs canonical single-source BEM.

A single-aperture coupled solve with U = A * v_unit must produce the
same pressure field as the equivalent canonical
``hornlab_solver.solve_frequencies`` call with
``velocity_sources={tag: v_unit}, velocity_mode=VELOCITY``.

This validates that the LEM->BEM coupling layer reduces correctly to
the canonical single-source case (i.e. introduces no computational
error of its own in the degenerate case).

Marked ``slow`` -- requires a working OpenCL CPU runtime and runs a
real BEM solve. Skip via ``pytest -m "not slow"``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_single_aperture_unit_velocity_matches_canonical():
    """U = A * v_unit -> coupled run == direct BEM with v_n = v_unit."""
    gmsh = pytest.importorskip("gmsh")
    pytest.importorskip("bempp_cl")
    pytest.importorskip("pyopencl")

    from hornlab_sim.methods import lem_to_bem
    from hornlab_solver import SolveConfig, solve_frequencies
    from hornlab_solver.config import BIEFormulation, LinearSolver, VelocityMode
    from hornlab_solver.mesh import load_mesh

    # --- Build a small closed sphere mesh with tag 2 = source --------
    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "sphere.msh"
        _build_sphere(mesh_path, radius_m=0.05, elem_size_m=0.025)

        # Solver config: canonical narrow-slot defaults + LU
        config = SolveConfig(
            formulation=BIEFormulation.COMPLEX_K,
            solver=LinearSolver.LU,
            freq_count=1,
            freq_min_hz=500.0,
            freq_max_hz=500.0,
        )

        freqs = np.array([500.0, 1000.0])

        # Compute aperture face area so we can pick U = A * v_unit
        loaded = load_mesh(mesh_path)
        from hornlab_sim.methods.lem_to_bem import _aperture_face_areas

        area_m2 = _aperture_face_areas(loaded, {"src": [2]})["src"]

        # Pick a non-trivial unit velocity so any scaling bug shows up
        v_unit = 0.123 + 0.0j
        U = np.full(len(freqs), area_m2 * v_unit, dtype=np.complex128)

        # --- 1. lem_to_bem path ---
        result_coupled = lem_to_bem.solve(
            mesh=loaded,
            lem_velocities={"src": U},
            aperture_tags={"src": [2]},
            frequencies_hz=freqs,
            config=config,
        )

        # --- 2. Canonical direct BEM with the same v_n ---
        from dataclasses import replace

        direct_config = replace(
            config,
            velocity_mode=VelocityMode.VELOCITY,
            velocity_sources={2: complex(v_unit)},
        )
        result_direct = solve_frequencies(loaded, freqs, direct_config)

        # --- Compare ---
        # Complex pressure should match to within numerical noise.
        # Tolerance ~ float32 epsilon * a few; the solve is run in
        # single precision by default.
        np.testing.assert_allclose(
            result_coupled.pressure_complex,
            result_direct.pressure_complex,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Coupled lem_to_bem pressure differs from canonical BEM",
        )
        np.testing.assert_allclose(
            result_coupled.impedance,
            result_direct.impedance,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Coupled lem_to_bem impedance differs from canonical BEM",
        )


def _build_sphere(path: Path, radius_m: float, elem_size_m: float) -> None:
    """Closed sphere mesh, single surface physical group tag 2."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("parity_sphere")
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
