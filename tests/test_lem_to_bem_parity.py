"""Degenerate-case parity test: lem_to_bem.solve vs canonical single-source BEM.

A single-aperture coupled solve with U = A * v_unit must produce the
same pressure field as the equivalent canonical
``hornlab_bempp_bem.solve_frequencies`` call with
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
    from hornlab_bempp_bem import SolveConfig, solve_frequencies
    from hornlab_bempp_bem.config import BIEFormulation, LinearSolver, VelocityMode
    from hornlab_bempp_bem.mesh import load_mesh

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


@pytest.mark.slow
def test_multi_aperture_two_source_solve_completes():
    """Drive two independent apertures with different complex U(f).

    Smoke verification that the multi-aperture path is wired correctly:
    two distinct physical groups on the same mesh, each given its own
    complex U(f), produce a finite non-zero pressure field. The test
    does not assert a specific physical result (the geometry is
    contrived) -- it asserts that the solve completes, that the result
    pressure shape is right, and that the per-aperture v_n values were
    recorded in solver_log.
    """
    gmsh = pytest.importorskip("gmsh")
    pytest.importorskip("bempp_cl")
    pytest.importorskip("pyopencl")

    from hornlab_sim.methods import lem_to_bem
    from hornlab_bempp_bem import SolveConfig
    from hornlab_bempp_bem.config import BIEFormulation, LinearSolver
    from hornlab_bempp_bem.mesh import load_mesh

    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "two_patch_sphere.msh"
        _build_sphere_with_two_patches(mesh_path, radius_m=0.05, elem_size_m=0.025)

        loaded = load_mesh(mesh_path)
        tags_present = sorted(set(int(t) for t in loaded.physical_tags))
        assert 2 in tags_present and 3 in tags_present, (
            f"Two-patch sphere should have tags 2 and 3 (got {tags_present})"
        )

        config = SolveConfig(
            formulation=BIEFormulation.COMPLEX_K,
            solver=LinearSolver.LU,
        )

        freqs = np.array([500.0, 1000.0])
        U_a = np.array([1e-3 + 0j, 2e-3 + 0j])
        U_b = np.array([0.5e-3 + 0.5e-3j, 1e-3 - 0.5e-3j])

        result = lem_to_bem.solve(
            mesh=loaded,
            lem_velocities={"patch_a": U_a, "patch_b": U_b},
            aperture_tags={"patch_a": [2], "patch_b": [3]},
            frequencies_hz=freqs,
            config=config,
            area_tolerance=1.0,  # patches are intentionally asymmetric
        )

        # Pressure is non-degenerate
        assert result.pressure_complex.shape[0] == 2  # n_freq
        assert np.all(np.isfinite(result.pressure_complex))
        assert np.any(np.abs(result.pressure_complex) > 0)

        # solver_log captured the per-aperture v_n values per frequency
        lem_logs = [
            entry["lem_to_bem"]
            for entry in result.solver_log
            if isinstance(entry, dict) and "lem_to_bem" in entry
        ]
        assert lem_logs, "Expected lem_to_bem entry in solver_log"
        assert len(lem_logs[0]) == 2  # one record per frequency
        for rec in lem_logs[0]:
            assert "patch_a" in rec["v_n_per_aperture"]
            assert "patch_b" in rec["v_n_per_aperture"]


def _build_sphere_with_two_patches(path: Path, radius_m: float, elem_size_m: float) -> None:
    """Sphere meshed with two physical groups (upper hemisphere = 2, lower = 3).

    Splits the sphere surface at z=0 via a plane cut so the two patches
    have distinct physical-group tags and substantial face area.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("two_patch_sphere")
        sphere = gmsh.model.occ.addSphere(0, 0, 0, radius_m)
        # Cut the sphere with the z=0 plane to get two hemispherical
        # volumes -> their boundary surfaces are upper/lower hemispheres.
        box_lower = gmsh.model.occ.addBox(
            -2 * radius_m, -2 * radius_m, -2 * radius_m,
            4 * radius_m, 4 * radius_m, 2 * radius_m,
        )
        upper, _ = gmsh.model.occ.cut([(3, sphere)], [(3, box_lower)], removeTool=False)
        gmsh.model.occ.synchronize()

        # Re-add full sphere for lower
        sphere2 = gmsh.model.occ.addSphere(0, 0, 0, radius_m)
        box_upper = gmsh.model.occ.addBox(
            -2 * radius_m, -2 * radius_m, 0,
            4 * radius_m, 4 * radius_m, 2 * radius_m,
        )
        lower, _ = gmsh.model.occ.cut([(3, sphere2)], [(3, box_upper)], removeTool=True)
        gmsh.model.occ.synchronize()

        # Collect surfaces (boundaries) of both volumes
        upper_surfs = gmsh.model.getBoundary(upper, oriented=False)
        lower_surfs = gmsh.model.getBoundary(lower, oriented=False)

        # Keep only curved (sphere) faces, not the flat z=0 cuts. The
        # flat face is the one with normal +/- z.
        def is_curved(dim_tag):
            # Heuristic: get the centre of mass; flat z=0 cut has com.z ~ 0.
            com = gmsh.model.occ.getCenterOfMass(*dim_tag)
            return abs(com[2]) > radius_m * 0.1

        upper_curved = [s for s in upper_surfs if is_curved(s)]
        lower_curved = [s for s in lower_surfs if is_curved(s)]

        # Remove the volumes; mesh only the curved surfaces
        gmsh.model.occ.remove(upper, recursive=False)
        gmsh.model.occ.remove(lower, recursive=False)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [s[1] for s in upper_curved], tag=2)
        gmsh.model.setPhysicalName(2, 2, "patch_a")
        gmsh.model.addPhysicalGroup(2, [s[1] for s in lower_curved], tag=3)
        gmsh.model.setPhysicalName(2, 3, "patch_b")

        gmsh.option.setNumber("Mesh.MeshSizeMax", elem_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", elem_size_m)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


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
