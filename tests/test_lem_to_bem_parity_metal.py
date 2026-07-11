"""Multi-source parity test for the native Metal LEM->BEM backend."""

from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_multi_aperture_superposition_matches_per_frequency_metal_loop():
    """Unit-basis superposition matches varying two-source direct solves."""
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
        mesh_path = Path(td) / "box.msh"
        _build_two_source_box(mesh_path, side_m=0.1, elem_size_m=0.035)

        config = metal.native_config(
            formulation="complex_k",
            freq_count=1,
            freq_min_hz=500.0,
            freq_max_hz=500.0,
        )

        freqs = np.array([500.0, 1000.0])
        loaded = load_mesh(mesh_path)
        aperture_tags = {"left": [2], "right": [3]}
        areas = _aperture_face_areas(loaded, aperture_tags)

        velocity_left = np.array([0.123 + 0.04j, -0.07 + 0.11j])
        velocity_right = np.array([-0.08 + 0.03j, 0.09 - 0.05j])

        result_coupled = lem_to_bem.solve(
            mesh=loaded,
            lem_velocities={
                "left": areas["left"] * velocity_left,
                "right": areas["right"] * velocity_right,
            },
            aperture_tags=aperture_tags,
            frequencies_hz=freqs,
            config=config,
        )

        direct_results = []
        for index, frequency in enumerate(freqs):
            direct_config = replace(
                config,
                velocity_mode=VelocityMode.VELOCITY,
                velocity_sources={
                    2: complex(velocity_left[index]),
                    3: complex(velocity_right[index]),
                },
            )
            direct_results.append(
                metal.solve_frequencies(loaded, [frequency], direct_config)
            )
        direct_pressure = np.concatenate(
            [result.pressure_complex for result in direct_results], axis=0
        )
        direct_impedance = np.concatenate(
            [result.impedance for result in direct_results], axis=0
        )
        directivity_db = np.concatenate(
            [result.directivity_db for result in direct_results], axis=0
        )
        direct_surface_pressure = {
            tag: np.concatenate(
                [result.surface_pressure_avg[tag] for result in direct_results]
            )
            for tag in (2, 3)
        }

        np.testing.assert_allclose(
            result_coupled.pressure_complex,
            direct_pressure,
            # GPU float32 accumulation order differs between the shared-assembly
                # multi-source path and separate per-frequency solves; ~1e-5 relative
                # noise is expected, so pin at 1e-4 (far below physical significance).
                rtol=1e-4,
            atol=1e-8,
            err_msg="Coupled lem_to_bem pressure differs from direct Metal BEM",
        )
        np.testing.assert_allclose(
            result_coupled.impedance,
            direct_impedance,
            rtol=1e-4,
            atol=1e-8,
            err_msg="Coupled lem_to_bem impedance differs from direct Metal BEM",
        )
        np.testing.assert_allclose(
            result_coupled.directivity_db,
            directivity_db,
            # dB values cross zero, so rtol alone is meaningless there;
            # 1e-4 dB absolute is far below any physical significance.
            rtol=1e-4,
            atol=1e-4,
            err_msg="Coupled lem_to_bem directivity differs from direct Metal BEM",
        )
        for tag in (2, 3):
            np.testing.assert_allclose(
                result_coupled.surface_pressure_avg[tag],
                direct_surface_pressure[tag],
                rtol=1e-4,
                atol=1e-8,
                err_msg=(
                    "Coupled lem_to_bem surface pressure differs from direct "
                    f"Metal BEM for tag {tag}"
                ),
            )


def _build_two_source_box(path: Path, side_m: float, elem_size_m: float) -> None:
    """Closed box with opposite source faces tagged 2 and 3."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("metal_parity_box")
        half = side_m / 2.0
        volume = gmsh.model.occ.addBox(
            -half,
            -half,
            -half,
            side_m,
            side_m,
            side_m,
        )
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getBoundary([(3, volume)], oriented=False)
        ordered_x = sorted(
            (surface[1] for surface in surfaces),
            key=lambda tag: gmsh.model.occ.getCenterOfMass(2, tag)[0],
        )
        left_tag = ordered_x[0]
        right_tag = ordered_x[-1]
        rigid_tags = [
            tag for tag in ordered_x if tag not in (left_tag, right_tag)
        ]
        gmsh.model.occ.remove([(3, volume)], recursive=False)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, rigid_tags, tag=1)
        gmsh.model.addPhysicalGroup(2, [left_tag], tag=2)
        gmsh.model.addPhysicalGroup(2, [right_tag], tag=3)
        gmsh.model.setPhysicalName(2, 1, "rigid")
        gmsh.model.setPhysicalName(2, 2, "left")
        gmsh.model.setPhysicalName(2, 3, "right")
        gmsh.option.setNumber("Mesh.MeshSizeMax", elem_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", elem_size_m)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
