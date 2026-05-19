"""Smoke example: lem_to_bem on a trivial mesh.

This builds a tiny in-memory mesh (a single triangle with a velocity-source
tag) and runs the coupling layer end-to-end with a 3-frequency sweep, then
prints the resulting on-axis SPL. Intended as the "does it import and run"
smoke test, not a physically meaningful simulation.

For a real Synergy/MEH workflow, see
MEH-Lab/scripts/synergy_directivity_example.py.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np


def _write_tiny_msh(path: Path) -> None:
    """Write a tiny gmsh v2.2 .msh with two triangles forming a unit square.

    Physical tag 1 = rigid (one triangle), tag 2 = velocity source (other).
    Used only for smoke testing the coupling layer's plumbing; the BEM
    result on a unit-square monopole-ish geometry is not physically
    meaningful, but the solver does run.
    """
    content = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "rigid"
2 2 "source"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 0.1 0.0 0.0
3 0.0 0.1 0.0
4 0.1 0.1 0.0
$EndNodes
$Elements
2
1 2 2 1 1 1 2 3
2 2 2 2 2 2 4 3
$EndElements
"""
    path.write_text(content)


def main():
    from hornlab_sim.methods import lem_to_bem

    freqs = np.array([200.0, 400.0, 800.0])
    # Tag 2 ("source") in the tiny mesh -> "throat" aperture
    aperture_tags = {"throat": [2]}
    # Unit volume velocity at all frequencies (LEM would produce this)
    lem_velocities = {"throat": np.ones(len(freqs), dtype=np.complex128) * 1e-3}

    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "smoke.msh"
        _write_tiny_msh(mesh_path)

        print(f"Mesh: {mesh_path}")
        print(f"Frequencies: {freqs}")
        print(f"Aperture velocities (m^3/s): {lem_velocities}")

        result = lem_to_bem.solve(
            mesh=mesh_path,
            lem_velocities=lem_velocities,
            aperture_tags=aperture_tags,
            frequencies_hz=freqs,
        )

        print()
        print(f"frequencies_hz:  {result.frequencies_hz}")
        print(f"pressure shape:  {result.pressure_complex.shape}")
        print(f"on-axis SPL dB:  {result.spl_db[:, 0, 0]}")
        print("lem_to_bem smoke: OK")


if __name__ == "__main__":
    main()
