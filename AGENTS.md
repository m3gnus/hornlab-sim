# hornlab-sim — Agent Instructions

> Parent rules: [../AGENTS.md](../AGENTS.md)
> Sibling rules: [../MEH-Lab/AGENTS.md](../MEH-Lab/AGENTS.md)

This package is the canonical home for lumped (LEM), transfer-matrix (TMM),
Helmholtz, and LEM↔BEM coupling simulators. There is **no global project
orchestrator**, **no project registry**, **no MEH intent table** in this
package. Read this file and pick the right reusable method.

Project-specific orchestration belongs in the consuming project. For MEH-Lab
studies, use `MEH-Lab/tools/meh_pipeline/` as the workflow layer and keep this
package limited to reusable physics methods.

## Decision tree

| Design question | Method | Module |
|---|---|---|
| Box tuning, Fb, Qtc, BP4/BP6S alignment | LEM bandpass | `hornlab_sim.methods.bandpass` (`bp4_sealed_rear`, `bp6s_dual_ported`) |
| Slot-pocket Helmholtz tuning (BIGMEH-style slot = cavity + baffled hole) | LEM Helmholtz | `hornlab_sim.methods.helmholtz` (`bigmeh_slot_helmholtz`, `bigmeh_mid_chamber_helmholtz_from_params`) |
| Mid-chamber resonance for a midport pocket | LEM Helmholtz | `hornlab_sim.methods.helmholtz` (`bigmeh_mid_chamber_helmholtz_from_params`) |
| Axial duct impedance with viscothermal losses (segmented horn/duct) | TMM | `hornlab_sim.methods.transfer_matrix` (`duct_input_impedance`, `uniform_tube_matrix`, `make_slot_tmm_load`) |
| Full 3D directivity from prescribed velocity sources (no LEM) | BEM | `hornlab_metal_bem.solve_frequencies` directly, or `MEH-Lab/tools/bigmeh_parametric/wg_bem.py` for BIGMEH cabinets |
| Full 3D directivity with realistic LEM/TMM-derived source velocities at apertures | Source-basis BEM coupling | `hornlab_sim.methods.lem_to_bem` for generic/custom meshes; `MEH-Lab/tools/bigmeh_parametric/source_basis_wg_bem.py` plus `lem_wg_bem.py` for BIGMEH validation heatmaps |
| Reduced FEM/BEM-style aperture back-loading | BEM radiation impedance matrix | `hornlab_sim.methods.radiation_impedance` |
| Voltage-driven coupled cone+port velocities for a BEM-terminated cardioid branch | LEM driver/BEM termination coupling | `hornlab_sim.methods.driver_coupling` |

## Canonical interpretations (load-bearing — do not change without explicit user sign-off)

- **Slot pocket = front cavity. Slot exit = baffled hole, L=0, end correction only.** This is the BIGMEH-canonical interpretation. Helmholtz code in this package implements it by default. Memory: `feedback_helmholtz_slot_pocket.md`.
- **Kirchhoff-Benade viscothermal losses** are built into `transfer_matrix.py`. Beranek & Mellow 2012 formulation. Don't replace with a different loss model without rerunning the Hornresp parity tests.
- **Bandpass `Port(Q_port=None)` derives viscothermal Q from geometry** using the same Kirchhoff-Benade boundary-layer scaling. Pass an explicit numeric `Q_port` for fixed-Q legacy behavior.
- **BP4/BP6S** are validated within ~2 dB vs Hornresp across the LF passband.

## LEM↔BEM coupling rules

When using `hornlab_sim.methods.lem_to_bem`:

1. **Suppress LEM-side radiation loading at any aperture that BEM is going to radiate.** For Helmholtz helpers, pass `end_corr="none"` for that aperture. For bandpass `Port`, pass `radiation_external=False` so both the outside end correction and radiation resistance are suppressed. Otherwise the radiation loading is double-counted (once by LEM, once by BEM-computed radiation impedance) and you get systematic dB errors at the LF/MF crossover.
2. **Aperture name → list of physical group IDs.** BIGMEH meshes already produce multiple tags per slot (exit + walls). The coupling layer expects `dict[str, list[int]]`, not 1:1.
3. **Area mismatch is a warning, not an error.** Sum of BEM face areas vs LEM-assumed S can disagree up to ~5% from mesh discretization. Warn, log to result metadata, continue.
4. **Velocity vs acceleration mode.** Canonical solver default is `velocity_mode=ACCELERATION`. LEM emits volume velocity U. The coupling layer converts U → v_n and lets the solver apply jω. Don't pass acceleration directly.
5. **Phase reference.** All LEM aperture velocities share an excitation reference (driver terminal voltage). Mesh-side BCs must use the same complex sign convention. The +iωρv convention matches `hornlab_metal_bem` canonical settings.

For BIGMEH/Synergy validation plots, use MEH-Lab's
`bigmeh_parametric.source_basis_wg_bem.run()` (or the LEM-specific
`lem_wg_bem.run()` wrapper) instead of calling this generic solver-level
coupling directly. Those adapters route the BEM solve through
`bigmeh_parametric.wg_bem.run`, which owns the BIGMEH observation frame,
basis caching, and canonical `results.npz`/heatmap schema.

## Solver settings (BEM side, from parent AGENTS.md §1)

- BM=off everywhere
- COMPLEX_K formulation; complex_k_shift=0.005 when enclosed
- LU solver for narrow slots, AUTO otherwise
- DP0/P1, q=4 regular quadrature
- +iωρv sign convention

These defaults live in `hornlab_metal_bem.SolveConfig`. Don't override without explicit reason.

## What lives where

| Concept | Location |
|---|---|
| LEM/TMM/Helmholtz core math | `hornlab_sim.methods.*` (this package) |
| BIGMEH cabinet geometry, params, slot interpretation knobs | `MEH-Lab/tools/bigmeh_parametric/` |
| BIGMEH → lumped adapter | `MEH-Lab/tools/lumped/from_bigmeh.py` |
| TMM CLI (BIGMEH-aware) | `MEH-Lab/tools/lumped/tmm_cli.py` |
| Aperture radiation impedance matrices | `hornlab_sim.methods.radiation_impedance` |
| BEM mesher | `hornlab-waveguide-mesher` sibling package |
| BEM solver | `hornlab-metal-bem` sibling package |
| BEM canonical caller for BIGMEH | `MEH-Lab/tools/bigmeh_parametric/wg_bem.py` |
| MEH project orchestration / tool registry | `MEH-Lab/tools/meh_pipeline/` |

If a new MEH project-specific calculation needs to be added, it goes in
MEH-Lab. If it's general lumped or coupling physics, it goes here.

## Auto-commit policy

Per parent `AGENTS.md`. Commit logical units without asking. Include doc updates in the same commit as the code change.
