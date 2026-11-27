# THeatA surface-code prior calibration

A modular experiment package for estimating **known graphlike DEM edge priors**
from detector covariances, followed by a paired MWPM logical-error-rate comparison.
The default experiment uses the rotated surface-code workload and LIZ
Predictive Trigger (LPT) of Figure 7 in THeatA. The standard Stim native
bridge is available for the existing direct-cooling sweeps. [THeatA](https://github.com/MSTakahashiPJ/theata)

The target of estimation is the identifiable matching graph obtained by
projecting the full hardware DEM onto the memory basis's detector sector.
The original full DEM is also saved. The package does **not** claim to infer
the separate probabilities of all full-DEM hyperedges from second moments.

## Quick start

Use Python 3.10 or newer on Linux, macOS, or WSL. From this directory:

```bash
bash scripts/bootstrap.sh
bash scripts/run.sh --config configs/lpt_calibration_smoke.yaml --out results/lpt_calibration_smoke --skip-saturated-marginals
```

The setup creates `.venv`, installs the package and notebook dependencies, and
checks out THeatA at tested commit
`8bfb9b97de91bacac2f56725c27b4c71c6a91009`. An existing checkout can be selected
without modifying it:

```bash
export THEATA_PATH=/absolute/path/to/theata
bash scripts/bootstrap.sh
```

With an existing Python environment:

```bash
python -m pip install -e '.[notebook,test]'
export THEATA_PATH=/absolute/path/to/theata
theata-prior run --config configs/lpt_calibration.yaml --out results/lpt_calibration --skip-saturated-marginals
```

`lpt_calibration.yaml` sweeps `liz_pt_thresholds` at distances 3, 5, and 7 with
three rounds and 100,000 heldout shots per condition. The smoke preset uses d=3,
threshold 0.2 and 4,000 heldout shots. Use a fresh output directory for each
scientific configuration. A matching directory resumes the saved run.

## Notebook

```bash
bash scripts/bootstrap.sh
.venv/bin/jupyter lab notebooks/calibration_walkthrough.ipynb
```

Select the **THeatA prior calibration** kernel. The notebook runs
`configs/lpt_calibration.yaml` and writes to `results/lpt_calibration` by
default. The first cell can override distances, fixed round count, LPT thresholds,
maximum iterations, cooling duration, residual phonons, calibration and test shots,
repetitions, workers, seed, and output directory. Set `RUN_EXPERIMENT = False`
to review a saved run.

The walkthrough shows full-detector and Z-sector LER over the complete experiment,
adjacent-distance suppression ratios, physical metrics, the calibration-shot
sweep, RMSE, and raw S11 heatmaps. The paper comparison uses the full-detector
LERs reported separately in `full_detector_metrics.csv`. `metrics.csv` stores
the paired sector comparison and `suppression_ratios.json` stores the ratios.

For scripted execution, run a copy of the notebook to keep its outputs
separate from the tracked source.

```bash
mkdir -p results
cp notebooks/calibration_walkthrough.ipynb results/lpt_walkthrough.ipynb
THEATA_PRIOR_CONFIG=lpt_calibration_smoke.yaml .venv/bin/python scripts/execute_notebook.py results/lpt_walkthrough.ipynb
```

The notebook records undefined S11 fits when any empirical detector mean
reaches 1/2. The CLI enables the same behavior with `--skip-saturated-marginals`
on `run`, or on both `fit` and `evaluate` for staged execution.

## Unknown detector supports

The separate structure benchmark learns singleton and pair detector supports
from calibration data. It compares complete-family zero clipping and statistical
selection with the actual and known-support references. The optional
calibration-observable stage learns logical labels for decoding.

```bash
bash scripts/run_structure.sh --config configs/structure_smoke.yaml --out results/structure_smoke
```

See the [method and executed smoke results](docs/unknown_structure.md) and the
[structure learning walkthrough](notebooks/structure_learning_walkthrough.ipynb).

## Decoder-specific prior training

The `decoder-training` workflow uses the Cross-Entropy Method to learn separate
grouped DEM priors for MWPM and ldpc Union-find. Detector supports and logical
actions are fixed. Validation selects checkpoints, and a shared test set
measures transfer across decoders. [Decoder training guide](docs/decoder_training.md)

```bash
.venv/bin/python -m pip install -e '.[decoder-training,notebook]'
scripts/run_decoder_training.sh --config configs/decoder_training_smoke.yaml --out results/decoder_training_smoke
```

## Thermal benchmarks

Run the THeatA Figure 4 and Figure 6 surface-code profiles and corresponding
triangular color-code experiments through the separate `thermal` commands.

```bash
bash scripts/bootstrap.sh --thermal
.venv/bin/theata-prior thermal run --config configs/thermal_surface_cap_smoke.yaml \
  --out results/thermal_surface_cap_smoke --workers 2 --resume
.venv/bin/jupyter lab notebooks/thermal_getting_started.ipynb
```

The [getting-started notebook](notebooks/thermal_getting_started.ipynb) runs
all four simulations and regenerates the figures. The [thermal methods](docs/thermal_benchmarks.md)
describe the physics, compiler, decoding, production presets, PBS commands and
executed validation.

## Standard Stim direct-cooling experiment

For each distance, round count, basis, and cooling threshold, the package runs these stages.

1. Generates the noiseless standard Stim rotated-memory circuit with
   `rounds = distance * rounds_per_distance`.
2. Expands H and CX into verified native Clifford identities and compiles the
   native sequence independently with the specified THeatA cooling threshold. The conservative
   total ordering retains all original data and ancilla qubits; it does not
   substitute THeatA's ancilla-reuse example circuits.
3. Inserts the THeatA MS Pauli lookup noise at each physical MS gate, one-qubit,
   reset and readout errors, and per-ion idle dephasing from the compiled
   schedule. THeatA's phonon tracking and scheduler timing are recorded.
4. Uses the 15 lookup probabilities directly in each native MS channel and
   applies the configured one-qubit, reset, readout, and dephasing parameters
   as given. Records the **mean total nonidentity Pauli probability per MS
   location** as the measured `mean_ms_error`.
5. Extracts the full DEM, then validates and constructs its graphlike marginal
   in the selected memory basis. A probability-free topology and an illustrative
   uniform-prior `placeholder.dem` are saved. The placeholder values are not
   used by the estimator.
6. Samples the full actual DEM, retaining the selected detector columns and all
   logical labels. Calibration and evaluation use distinct deterministic random
   streams. All N values use prefixes of one maximum-size calibration dataset.
7. Applies exact equation S11 to known pair edges and S14–S16 to existing
   boundary edges. Logical labels and actual priors are not inputs to fitting.
8. Decodes the same heldout shots with actual-sector and estimated-sector
   priors, using PyMatching `decode_batch`. Both decoders use weights
   `log((1-p_e)/p_e)` and the same graph and logical edge labels.

The default sampling source is `dem`. This makes the benchmark's true
generative model the independent Bernoulli DEM and isolates prior calibration.
`sample_source: circuit` samples the compiled noisy circuit instead. General
`PAULI_CHANNEL_2` conversion uses Stim's `approximate_disjoint_errors=True`;
in circuit mode, circuit-to-DEM approximation is an additional source of model
mismatch. The approximation is recorded, not hidden.

The inferred LER is the probability of any logical-observable mismatch per
**complete memory experiment**, not a rate per syndrome round.

## Stage-by-stage CLI and Python API

```bash
theata-prior build    --config configs/demo.yaml --out results/demo_cooling --workers 8
theata-prior sample   --config configs/demo.yaml --out results/demo_cooling --workers 8
theata-prior fit      --config configs/demo.yaml --out results/demo_cooling --workers 8 --skip-saturated-marginals
theata-prior evaluate --config configs/demo.yaml --out results/demo_cooling --workers 8 --skip-saturated-marginals
theata-prior plot --out results/demo_cooling
```

Each stage uses the preceding stage's saved files. `run` performs all stages.
Re-running the same experiment resumes completed artifacts. Scientific
configuration changes are rejected within an existing output directory;
worker counts may change. A completed model cache is reused as archived,
even if a different checkout path is supplied. Use a new output directory
when changing compiler source or model implementation.

```python
from theata_prior.config import load_config
from theata_prior.experiment import (
    build_models, sample_models, fit_models,
    evaluate_models_stage, run_experiment,
)
from theata_prior.plotting import plot_results

config = load_config("configs/demo.yaml", workers=8)
rows = run_experiment(config, "results/demo_cooling", skip_saturated_marginals=True)
```

For an already prepared identifiable graph:

```python
from theata_prior.dem import project_dem
from theata_prior.estimation import estimate_priors

projected = project_dem(actual_full_dem, detector_ids)
# Only the probability-free structure crosses the fitting API.
fit = estimate_priors(projected.structure, calibration_detector_array, workers=8)
estimated_dem = fit.to_dem()
fit.save("results/custom_fit")
```

## Configuration and CPU parallelism

Every YAML key is validated; unknown keys raise. Main fields are:

| Field | Meaning |
|---|---|
| `distances`, `basis`, `rounds_per_distance` | Standard Stim memory-code geometry |
| `rounds`, `liz_pt_thresholds` | Fixed rounds and the LPT threshold sweep for reconstructed Z memory |
| `liz_pt_max_iters`, `cooling_duration_us`, `residual_phonon` | LPT cycle budget and abstract cooling settings |
| `cool_thresholds` | Distinct nonnegative transverse phonon thresholds. Use `null` to disable pre-MS threshold cooling |
| `training_shots` | Strictly increasing detector-only calibration prefix sizes |
| `test_shots` | Independent heldout shots per compiled condition, shared by compared decoders |
| `repetitions` | Independent calibration streams; heldout data remain paired |
| `workers` | CPU processes/threads, or `auto` respecting PBS/SLURM allocation |
| `sample_chunk_shots` | Fixed sampling shard size; part of seed reproducibility |
| `decode_chunk_shots` | Heldout batch size per decoding task |
| `covariance_chunk_shots`, `covariance_block_detectors` | Bounded-memory covariance chunks and parallel row blocks |
| `probability_floor`, `probability_ceiling` | Fixed decoder-prior regularization, independent of actual priors |
| `strict_estimation` | Reject nonphysical finite-sample estimates instead of clipping |
| `sample_source` | `circuit` for LPT. The standard Stim sweep supports `dem` and `circuit` |
| `hardware` | Compiler, timing and noise overrides for the standard Stim sweep |

Sampling and decoding use process pools with the `spawn` start method, worker
local Stim/PyMatching objects, and disk-backed NumPy arrays. Covariance uses
float64 matrix products on shared input chunks and parallel detector-row
blocks. BLAS threads are restricted to one per worker to avoid oversubscription.
The memory cost of covariance is O(B M + M²), with B the chunk size and M the
selected detector count; no N-by-M float64 copy is retained. Small detector
matrices need fewer covariance workers than large ones.

Seeds are derived from a master seed, distance, rounds, basis, cooling threshold, train/test namespace,
calibration repetition, and shard ID. Changing worker count does not change
samples for fixed chunk size and Stim build. Stim does not guarantee bitwise
identity across versions or CPU SIMD architectures. The maximum N is sampled
once to ensure smaller N values are exact prefixes.

The sample arrays use one byte per detector/observable. Approximate disk size
per compiled condition is `(repetitions * max(training_shots) + test_shots) * (M+K)`
bytes, with K logical observables. The complete p_ij matrix requires O(M²)
storage. Files are memory-mapped; large studies do not load all samples in RAM.

For PBS/Torque, adapt the resource header in `scripts/run_pbs.sh`, then submit
from the project root with `qsub scripts/run_pbs.sh`. The provided example uses
`workq`, one node, and 96 CPUs.

## Standard Stim hardware options and recorded limits

The `hardware` mapping accepts `liz_capacity`, `memory_capacity`,
`compiler_options`, `noise_parameters`, `time_parameters`, `lookup_mode`,
`phonon_scale`, and `strict_lut`. Compiler dictionaries are passed to upstream
dataclasses and invalid options raise. The defaults use a single LIZ and a
conservative native order. This is a valid hardware-aware execution of the
standard circuit, not a claim of an optimized QCCD schedule.

Use `cool_thresholds` for the pre-MS policy. The loader rejects a simultaneous
`hardware.compiler_options.cool_threshold_transversal` entry, including an entry
set to `null`. Other hardware settings apply throughout the sweep. By default,
the memory and temporary zone capacity is sized for the largest distance and
reused across distances. The default `phonon_scale` is 1.0. The independent
structure-learning and decoder-training benchmarks retain their existing
`physical_error_rates` settings, and `plot` can still read legacy calibration
results.

Before each MS gate, the THeatA policy compares the largest transverse
occupation of its target ions with the threshold. If it exceeds the threshold,
THeatA directly cools the entire LIZ chain, which can contain ions carrying
quantum information. This describes the upstream model's assumption. The model
takes a fixed cooling time and fixed residual phonon numbers as separate
inputs, without a dynamic relation. The threshold is the only cooling input
that varies across this sweep. `null` disables this policy. Other cooling
policies remain off by default.

The standard Stim `demo` and `production` YAMLs use `strict_lut: true`.
A condition above the LUT maximum keeps its compiled hardware trace and reason
in `hardware.json` and `model_failed.json`. The run continues with the usable
conditions. The no-cooling conditions exceed the LUT maximum under the default
hardware. `smoke.yaml` explicitly uses `strict_lut: false` to exercise the
upstream bounded lookup as well. For this mode, the raw phonons, selected LUT
rows, and counts of out-of-range gates are recorded.

## Output layout

LPT runs use case directories such as `d3_r3_z/lpt_0.2`. They store the
reconstructed ideal circuit, full heldout samples, and `evaluation_full.json`
in addition to the sector files. A `projection_error` in `model.json` records
unsupported sector topology while the full-detector evaluation proceeds.
The following example shows the standard Stim layout.

```text
results/demo_cooling/
  run.json                             # Config, versions, run identity
  physical_metrics.csv / physical_metrics.json
  metrics.csv / metrics.json           # Per-model LER, Wilson intervals, prior RMSE
  paired_comparisons.csv               # Discordant logical failures and rate difference
  figures/
    ler_vs_cooling_threshold.png/.pdf
    prior_rmse_vs_cooling_threshold.png/.pdf
    ler_vs_calibration_shots_d3.png/.pdf
    ler_vs_mean_ms_error_d3.png/.pdf
    ler_vs_execution_time_d3.png/.pdf
    priors_d3.png/.pdf                  # p_ij and actual/fitted prior comparison
  d3_r3_z/cool_0.25/
    stim_standard_ideal.stim
    theata_native_ideal.stim
    actual_noisy.stim
    actual_full.dem / actual_full_decomposed.dem
    actual_sector.dem / actual_priors.npy
    known_structure.json / placeholder.dem
    hardware.json / model.json
    samples_train_000/{detectors.npy,observables.npy,samples.json}
    samples_test/{detectors.npy,observables.npy,samples.json}
    fits/rep000_n1000/
      estimated.dem / estimated_priors.npy / raw_estimated_priors.npy
      p_ij.npy / moments.npz / boundary_probabilities.npy
      diagnostics.json / fit_complete.json
      prior_comparison.csv / prior_error.json
    evaluation.json
```

Raw S11 estimates preserve negative values and invalid-pair NaNs; only the
decoder's known edge priors are regularized. Boundary priors are computed
only for known boundary edges. Residual hyperedges, invisible logical events,
or one detector support with conflicting logical labels raise a structural
error. Detector marginals at or above 1/2 raise rather than being hidden by
clipping. Exact population-moment tests establish recovery under the model;
finite-sample clipping introduces bias that the saved diagnostics expose.

The physics table records the number and summed duration of cooling operations,
scheduled wall time, transverse phonon and MS error ranges, idle statistics,
and lookup status. For the phonon statistics, each gate contributes the maximum
transverse occupation of its two target ions at the gate. The `cool_off` case
uses JSON `null` and a separate plot category. Legacy result directories keep
their original paths and run fingerprints.

Figures show actual-prior and estimated-prior logical error rates against
cooling threshold, measured mean MS error, wall time, and calibration size. Wilson intervals condition on each fitted
model. When calibration is repeated, curves show repeat means and the envelope
of the individual intervals; repeated evaluations on the same test samples
are never pooled into fictitious independent trials. Zero-error results remain
zero in CSV and appear as upper-bound symbols on logarithmic plots.

## Verification and mathematical reference

```bash
THEATA_PATH=/absolute/path/to/theata .venv/bin/python -m pytest -q
```

Tests cover exact population recovery by independent Bernoulli enumeration,
S14–S16 boundaries, `^` correlated-event handling, logical-label ambiguity,
integer-overflow avoidance, worker-invariant sampling/decoding, resume behavior,
independent calibration/test streams, native Clifford equivalence, and X/Z
surface-code detector determinism. Integration tests require the pinned
upstream checkout.

See [docs/method.md](docs/method.md) for precise assumptions and equations.
The [general mechanism-prior derivation](docs/general_prior_derivation.md)
uses one mechanism index e and derives the real linear inverse problem for
arbitrary known detector supports, including its identifiability condition.
All package documentation, source comments, configuration text and notebook
text are in English. The general derivation does not change the executable
graphlike estimator.
Primary references are [Chen et al., arXiv:2102.06132, Supplement VI.1](https://arxiv.org/html/2102.06132v1),
the [Stim DEM specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md),
and [PyMatching documentation](https://pymatching.readthedocs.io/en/stable/).

