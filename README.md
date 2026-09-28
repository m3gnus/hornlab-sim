# hornlab-sim

Reusable lumped, transfer-matrix, Helmholtz, and LEM-BEM coupling
simulators for acoustic design work.

This package holds small/fast acoustic models that can be used standalone or
alongside the canonical Metal BEM pipeline.

## Modules

- `hornlab_sim.methods.bandpass` — BP4/BP6S lumped enclosure simulation, validated within ~2 dB vs Hornresp
- `hornlab_sim.hornresp` — Hornresp text config/response parsing, export, and validation helpers
- `hornlab_sim.methods.transfer_matrix` — segmented TMM with Kirchhoff-Benade viscothermal losses (Beranek & Mellow 2012)
- `hornlab_sim.methods.helmholtz` — cavity/aperture Helmholtz resonance helpers with configurable end correction
- `hornlab_sim.methods.max_spl` — Xmax- and voltage-limited SPL helpers
- `hornlab_sim.methods.bass_reflex` — WinISD-style bass-reflex seed and short-port screening core
- `hornlab_sim.methods.lem_to_bem` — forward LEM → BEM coupling (LEM-computed aperture velocities drive a BEM Neumann boundary condition)

## CLI

```bash
hornlab-tmm duct --width 130 --height 376 --depth 500 --termination rigid
hornlab-validate-hornresp CONFIG.txt DATA.txt --out /tmp/hornresp-compare.png
```

## Install

```bash
pip install "hornlab-sim @ git+https://github.com/m3gnus/hornlab-sim.git"

# Local development checkout:
git clone https://github.com/m3gnus/hornlab-sim.git
cd hornlab-sim
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests          # 162 tests: 160 pass, 2 skip
```

`[dev]` installs the `[fem]` extra too, because `tests/test_acoustic_fem.py`
imports the FEM module directly and fails — it does not skip — without SciPy.

**Two tests skip on a plain checkout.** Both need something that is not a
dependency of this package:

| Test | Needs | Install |
|---|---|---|
| `tests/test_lem_to_bem_parity_metal.py` | `gmsh` | `pip install gmsh` |
| `tests/test_lem_to_bem_unit.py::test_native_pure_grid_three_triangles_and_three_vertices` | the Metal solver | `pip install -e ".[metal]"` |

They cover the LEM→BEM seam, where the time-convention rule below is enforced
— which is the part of this package a single-aperture magnitude check cannot
see at all. Run them before changing anything at that seam. A skipped test and
a deleted one look identical in a green summary, so check the count: a run that
reports more than 2 skipped is measuring less than you think.

CI (`.github/workflows/ci.yml`) runs the plain install on Linux, macOS and
Windows and fails unless exactly those two tests skipped. A separate macOS job
installs `gmsh` and the `[metal]` extra and fails on any skip, so the seam tests
run on every push. `scripts/assert_skips.py` does the checking; update its
expected list in the workflow when a skip is added or removed on purpose.

## Agent / user guidance

See `AGENTS.md` for the decision tree: which method to reach for given the
design question being asked.

For LEM→BEM coupling, suppress LEM-side external radiation loading on any
BEM-radiated aperture. Use `end_corr="none"` with Helmholtz helpers and
`Port(..., radiation_external=False)` with bandpass ports.

`lem_to_bem.solve` takes engineering `e^{+j*omega*t}` volume velocities by
default and converts them once to the solver's `e^{-i*omega*t}` convention.
A caller that already holds solver-convention source phasors must say so with
`velocity_convention="solver"`; passing them as the default conjugates them a
second time and reverses the complex source phase.

## License

AGPL-3.0-or-later
