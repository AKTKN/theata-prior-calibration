# Thermal and cooling benchmarks

The four benchmark presets estimate the probability of a logical failure in a
complete memory experiment. Each shot samples the compiled physical Stim circuit.
The decoder is built from that condition's noisy circuit. The three syndrome
rounds in each preset are independent of the code distance.

## Workloads and thermal model

The surface presets follow the Figure 4 and Figure 6 workloads in THeatA
(arXiv:2611.30676v1). They use logical Z memory, the upstream distance-specific
circuits with one reused syndrome ancilla, greedy alternating initial placement,
and forward and reverse stabilizer order on alternating rounds. THeatA planner
v1 targets one LIZ of capacity four in M-M-M-L-M-M-M. The surface adapter
reconstructs CX gates with the upstream Approach B noise injection and computes
idle dephasing on the upstream per-ion lockstep timeline. The recompiler supplies
the terminal data measurements and trailing idle. An ordered operand and count
check aligns the reconstructed CX gates to the physical MS events before noise
injection. Surface decoding uses PyMatching with Stim's decomposed DEM.
[THeatA paper](https://arxiv.org/html/2611.30676v1) ·
[THeatA revision 8bfb9b9](https://github.com/MSTakahashiPJ/theata/tree/8bfb9b97de91bacac2f56725c27b4c71c6a91009)

The color presets use the triangular `tri_optimal` circuit, with
`superdense_circuit=False`, a noiseless source, and the Z memory boundary by
default. The color adapter retains all data and syndrome ions and compiles the
source's gate order through a total-order native DAG. It preserves measurement
indices, record offsets, observable definitions, and the full five-component
detector coordinates. The native rewrite uses H, M, R, H for MRX before expanding
the H gates into rotations. Native MS noise acts immediately after SQRT_XX and
idle dephasing acts in the physical basis before each native operation where it
can affect the circuit. This scheduling and noise placement define the color
extension and differ from the surface paper workload. The color X boundary is
also supported. [color-code-stim revision 0eb3593](https://github.com/seokhyung-lee/color-code-stim/tree/0eb35935c1e5ff30ba3db9def30a9d35bca2f16d)

Both adapters carry between-round ion placement and within-zone ordering. The
`reset_each_round` boundary resets motional occupations to zero at the start of
each round, following the paper's approximation. The wall clock stays
continuous. Color data readout stays in the last round's thermal segment. The
serial planning clock advances the phonons. The scheduled lockstep wall clock
determines idle times. Mainz axial and transverse background rates are 5.5, 4.8,
and 3.6 quanta/s. The defaults use $p_R=10^{-4}$,
$p_{\mathrm{meas}}=3.2\times10^{-3}$, $p_{\mathrm{reset}}=8\times10^{-4}$, and
$\gamma_\phi=3\,\mathrm{s}^{-1}$. All physical channels use unit scaling.

Each MS gate uses the hotter participant's first transverse occupation and the
upward-rounded row of the upstream 15-component lookup table. The cap sweep
compiles one no-cooling trace per distance and uses
$\bar n_{\mathrm{used}}=\min(\bar n_{\mathrm{raw}},\bar n_{\mathrm{cap}})$ for
the lookup. The table spans 0.01–20 quanta. Cap 0 selects the coldest row and
retains SPAM and idle dephasing. The named uncapped control uses raw phonons as
lookup input. Both controls and capped cases still obey the LUT range. Raw and
used clipping counts are recorded separately. The former describes where the
hardware trace exceeds the LUT support. The latter describes where the actual
lookup saturates. The production grid is a proposed reproduction sampling grid.
The original Figure 4 raw-data grid was unavailable.

Physical changes to heating accumulate during compilation. A constant
lookup-input experiment would substitute one occupation at every MS gate.
A cooling residual is a physical occupation floor after a CoolOp. The trigger
threshold is a predictor input and may be exceeded at a realized gate. The
implemented sweeps vary the virtual cap and MPT loop budget.

## MPT and decoder audit

The cooling presets compile no cooling, post-measurement LIZ cooling only, and
MPT at each positive budget $N_{\max}$. The MPT arms use THeatA's two-pass
measurement-window delta predictor, threshold 1.5 quanta, and the just-measured
syndrome ancilla as the actuator. The loop budget is per checkpoint. Each
completed loop can visit several memory zones. Physically cooling a LIZ chain
requires every occupant to be a measured, disposable syndrome ion. The adapter
tracks ion roles explicitly and checks occupants at each CoolOp. Gate or reset
operations end the ion's disposable lifetime. Final data measurement remains
data readout.

The preset uses 1000 μs for `TimeParameters.cool` and 1000 μs for
`cool_ancilla_after_measure_us`, with zero residual occupation across all
three modes. This follows the main-text 1 ms setting and the pinned
`exp_merge_count_diag.py` MPT setup. Appendix A gives 400 μs for generic cooling
and 2000 μs for deep recooling. The latter is also the upstream generic scan CLI
default. Each checkpoint audit records the next measurement window, target
ions, predicted gate occupations from the reference pass and realized gate
occupations. It also records activation and episode counts, memory visits,
CoolOp occupants, operation counts, and durations. The scheduler includes the
time and heating effects of added transport and cooling.
[Main-text cooling study](https://arxiv.org/html/2611.30676v1#S5.SS5) ·
[Pinned cooling diagnostic](https://github.com/MSTakahashiPJ/theata/blob/8bfb9b97de91bacac2f56725c27b4c71c6a91009/experiments/exp_merge_count_diag.py)

With `tri_optimal`, each round completes its CNOT layers before the first
syndrome measurement. No subsequent same-round measurement window contains an
MS gate. The last syndrome checkpoint can see the next round's first window in
the audit. Its initial phonons reset, so it is ineligible for this paper policy.
The budget arms therefore report `policy_inactive`, with zero eligible
checkpoints and zero MPT episodes. They share the post-measurement LIZ-only
physical circuit and use independent shot streams. In the no-cooling control
the post-measurement CoolOps are absent.

The color generator requests `exclude_non_essential_pauli_detectors=True` for
one memory observable. The original Tanner graph and five detector coordinates
are used by the color decoder. The adapter passes the flattened independent
Stim DEM of the actual noisy circuit directly to a `DemManager` and then to
`ConcatMatchingDecoder`. This preserves the chosen detector marginal through
the native rotations. Stim's `approximate_disjoint_errors=True` approximates
mutually exclusive Pauli-channel outcomes as independent DEM events for both
decoders. Shot sampling retains the exclusive outcomes from the physical
circuit. The color audit checks the actual error maps for each color and
reports discarded mechanism indices and their summed DEM probabilities. The
runner requires zero pruning in every color before sampling a condition.

## Commands

Use Python 3.11 or newer for the optional color package. Running the bootstrap
without the flag installs the existing calibration workflows with their current
dependency set.

```bash
bash scripts/bootstrap.sh --thermal
.venv/bin/jupyter lab notebooks/thermal_getting_started.ipynb

for family in surface_cap surface_cooling color_cap color_cooling; do
  .venv/bin/theata-prior thermal run --config "configs/thermal_${family}_smoke.yaml" \
    --out "results/thermal_${family}_smoke" --workers 4 --resume
done

.venv/bin/theata-prior thermal plot --out results/thermal_surface_cap_smoke
```

Each production preset uses 100,000 shots per point. The cap presets cover
distances 3 through 13 in odd steps, plus the uncapped control at each distance.
The cooling presets cover 3, 5, and 7 with $N_{\max}=1,2,4,8,16$ plus both
controls. To run locally, remove `_smoke` from the config and output names. To
submit one job per family from the repository root on a 96-core workq node, use

```bash
for family in surface_cap surface_cooling color_cap color_cooling; do
  qsub -v "THERMAL_CONFIG=configs/thermal_${family}.yaml,THERMAL_OUT=results/thermal_${family}" \
    scripts/run_thermal_pbs.sh
done
```

`thermal build` accepts the same options as `thermal run` and stops after model
construction. Follow it with `thermal run --resume` to sample and decode.
`--shots`, `--shard-shots`, and `--seed` override the preset and require a fresh
output directory when they change. `--workers` can change on resume and respects
the CPU allocation. The bounded spawn pools compile models and decode fixed
shot shards in separate stages, using a single process budget and numerical
library threads set to one. Each shard seed depends on the master seed,
scientific point identity, and shard index. Results reproduce for the same
Stim build and CPU SIMD environment. An atomic shard file records decoded
counts so resume rebuilds only missing shards. Git revisions, package versions,
implementation hashes, circuit/DEM/trace hashes, and all resolved settings are
recorded under the run directory.

`time_parameters` maps directly to the pinned `TimeParameters` fields. Durations
such as `ms_gate`, `split`, and `cool` are in μs. `shuttle_speed` is in μm/μs,
`shuttle_base_distance` in μm, and `shuttle_ion_factor` in μm per ion.
`heating_rate_*_per_s` fields are in quanta/s.
`phonon_shuttle_axial_per_180um` is quanta per 180 μm.
`phonon_shuttle_transversal`, `phonon_split_*`, `phonon_merge_*`,
and `phonon_swap_*` are per-operation increments in quanta. The
`phonon_cool_residual_*` fields and `cooling_threshold` are in quanta.
`post_measure_cooling_us` sets the actual direct-cooling duration in these
profiles, including the MPT actuator's CoolOps. The resolved defaults, including
the second-transverse fallbacks, are in `run.json`. `noise_parameters` accepts
the three discrete probabilities and `gamma_phi_per_s`.

`metrics.csv` and `metrics.json` hold integer counts, per-shot logical error
probabilities, and 95% Wilson intervals. A log-scale triangle marks the upper
bound for zero failures. `ratios.csv` gives $\Lambda=p_L(d)/p_L(d+2)$ with
conservative 95% bounds obtained from two 97.5% Clopper–Pearson intervals by
Bonferroni. A zero observed denominator gives an undefined ratio estimate and
an unbounded upper endpoint. A sampled threshold bracket requires opposite
pointwise confidence verdicts at adjacent caps. There is no interpolation or
simultaneous coverage claim across the grid. The paper uses labeled one-sigma
intervals for some figures. These plots use 95% intervals.

## Executed validation

The [getting-started notebook](../notebooks/thermal_getting_started.ipynb)
contains the four executed figures. The [smoke results](thermal_smoke_results.csv)
give the corresponding per-point counts. The checked environment used Python
3.12.7, Stim 1.16.0, PyMatching 2.4.0 and color-code-stim 1.1.1 at the pinned
revision.

- The 54 notebook points use three Z-memory rounds, 768 shots per point, and 192
  shots per shard. Both cap sweeps use caps 0, 1.5, 4, and 20 and the uncapped
  control. Surface runs use distances 3 and 5. Color runs use 3, 5, and 7.
  The surface cooling sweep also uses 8 and 16 cycles. The color cooling sweep
  uses 1, 2, and 4 cycles. Both include the two stated controls.
- With surface $d=3$, increasing the budget from 1 to 2 changed realized loop
  counts from 12 to 23 and total schedule time from 149.4 to 174.3 ms. For
  $d=5$ the same change gave 57 to 114 loops and 658.4 to 836.1 ms. At
  $d=3$, going from 8 to 16 gave just three additional loops across the
  whole experiment. Increasing any color budget executed zero additional MPT
  loops. Color post-measurement cooling still executed 18, 54, and 108
  direct CoolOps at $d=3,5,7$, respectively.
- Regression checks compared uncapped surface d=3 against the upstream noisy
  circuit, checked exact Clifford and MRX stabilizer flows and measurement
  records, and tested color detector metadata, zero pruning, and X memory at
  d=3. A controlled interruption after one decoded shard resumed with two
  workers and reproduced the serial counts without rewriting that shard.
- A separate 16-shot surface check exercised every cooling condition at
  $d=3,5,7$. The large-distance cap checks used 16 shots per point, caps
  0, 4, and 20, and the uncapped control at surface $d=7,9,11,13$ and
  color $d=9,11,13$. The large color checks retained every mechanism in each
  decoder color.
- A clean bootstrap installed the pinned color package and ran the color cap
  smoke grid at 16 shots per point. A local PBS-launcher check used surface
  $d=3$, caps 0 and 4 plus the uncapped control, 36 shots per point, and two
  workers. The PBS job itself requires a workq allocation.

The benchmark implementation was based on repository revision `d0c2253`.
The validation runs establish pipeline behavior and measured cooling activity.
The production sweeps and statistical reproduction of the published error rates
remain unexecuted.
