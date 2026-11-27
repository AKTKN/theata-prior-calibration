# Grouped decoder-prior training

The experiment fits a known-support covariance baseline and uses the Cross-Entropy
Method (CEM) to learn a fixed effective prior vector for each target decoder,
training budget, data replicate, optimizer seed, and initialization. It compares
actual canonical DEM priors, covariance priors, the realized initialization, and
both trained priors on the same held-out shots with each evaluation decoder.

## Install and run

From the repository root, bootstrap the tested THeatA checkout and the Python
environment, then install the decoder extra. The bootstrap uses THeatA revision
`8bfb9b97de91bacac2f56725c27b4c71c6a91009`.

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m pip install -e '.[decoder-training,notebook,test]'
scripts/run_decoder_training.sh --config configs/decoder_training_smoke.yaml --out results/decoder_training_smoke
```

The smoke defaults use distance 3, three rounds, memory Z, physical error
probability 0.001, 2,048 calibration shots, 2,048 additional validation shots,
and 4,096 test shots. Candidate evaluation uses two workers when the CPU
allocation allows it. The research YAML uses distances 3 and 5, three rates,
three calibration budgets, and larger validation and test samples. Increase
`physical.repetitions` to measure data-replicate variation and add values to
`optimizer_seeds` to measure optimizer variation. Rare failures may require
more validation and test shots.

```bash
.venv/bin/python -m theata_prior decoder-training preflight \
  --config configs/decoder_training_research.yaml --out results/decoder_training_research
scripts/run_decoder_training.sh --config configs/decoder_training_research.yaml --out results/decoder_training_research
```

Use the same config and output path for `build`, `sample`, `fit`, `train`,
and `evaluate` to execute stages individually. Preflight reports the full
mechanism and group counts after `build`. Use `preflight --probe-shots 128`
after `sample` for a serial training estimate based on decoding real training
shots. The estimate scales a single uniform-model probe and varies with
syndrome density, candidate weights, and hardware. Its shot budget includes
validation and allows two extra controls per generation. Build, process
startup, I/O, and final evaluation are outside the estimate.

Rerun the shell command or pass `--resume` to `decoder-training run` to
continue. An interrupted training run resumes at the last committed
generation. Execution resources and covariance or decoding chunk sizes may
change on resume. Changes to scientific inputs require a fresh output path.
Sample chunk size is scientific because it determines the Stim shard seeds
and sample realization. `--no-resume` requires a fresh directory.

```bash
.venv/bin/python -m theata_prior decoder-training plot \
  --out results/decoder_training_smoke --distance 3 --physical-error-rate 0.001 \
  --calibration-shots 2048 --replicate 0 --formats png pdf
.venv/bin/jupyter lab notebooks/cem_decoder_prior_calibration.ipynb
```

In Jupyter, select the **THeatA prior calibration** kernel installed by
bootstrap. The first code cell exposes `target_decoders`, `evaluation_decoders`,
and `plot_decoders`. Use `("uf",)`, `("mwpm",)`, or `("mwpm", "uf")`.
`None` uses the YAML setting. The same keys can be set in YAML, for example
`target_decoders: [uf]`, `evaluation_decoders: [uf]`, and `plot_decoders: [uf]`.

Running with `stages = ("plot",)` loads and replots existing results.
`plot_decoders` selects evaluation panels and corresponding trained priors.
The figures include the actual and covariance baselines. A single-decoder
selection writes to `figures/mwpm` or `figures/uf`. With the default null
`plot_decoders`, the notebook shows every saved decoder. The plot CLI accepts
`--decoders uf`, `--decoders mwpm`, or `--decoders mwpm uf`.

For a headless execution from the repository root, run the notebook on a copy
with the same environment.

```bash
mkdir -p results
cp notebooks/cem_decoder_prior_calibration.ipynb results/decoder_training_execution.ipynb
MPLBACKEND=Agg .venv/bin/python scripts/execute_notebook.py results/decoder_training_execution.ipynb
MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/test_decoder_training.py
```

The notebook calls `theata_prior.decoder_training.experiment` stages directly.
`config.load_training_config` loads YAML, `config.config_from_dict` validates
in-memory settings, `cem.optimize` trains one fixed-structure model, and
`plotting.load_results` and `plotting.plot_results` work from saved tables.

## Information and sample protocol

The physical pipeline compiles the standard rotated Stim surface-code circuit
through THeatA and projects onto the selected memory sector. The fixed
detector incidence matrix H and logical-action matrix B follow the canonical
`GraphStructure.edges` order. A column has one or two detector targets.
Repeated identical mechanisms are XOR-aggregated by the existing DEM
projection. Source detector IDs and coordinates are preserved. Canonical
actual probabilities describe those mechanisms.

Each physical case has a maximum training pool and an independent validation
set per data replicate. Calibration budgets select nested prefixes of that
pool, and both decoders train on the same prefix. Covariance fitting uses
training detections D only. CEM uses observed D and logical outcomes L.
Validation D,L select among the initialization and checkpoint candidates.
One additional test stream per physical case supplies D,L for all frozen
comparisons. The optimizer signature receives only the known structure,
initialized priors, groups, training and validation sample references, and
optimization settings. Data seeds and optimizer seeds use separate stable
namespaces. The default source is the full actual DEM sampled and projected to
the selected detectors. Set `physical.sample_source: circuit` for the actual
noisy-circuit sampling path.

## Groups, scale and optimization

The default coarse grouping uses singleton or pair support, then classifies
pair displacement by same site, axial, or diagonal spatial displacement and
by whether the round changes. An edge that touches the first observed round
is initial. Otherwise an edge that touches the last round is final. The
remaining edges are bulk. The experiment verifies coordinates against the
standard Stim circuit before using them. The support-count fallback applies
to graphs with unverified or missing coordinates. To specify groups directly,
set `grouping: manual` and supply `manual_groups` as a list of labels aligned
to canonical edges, or as a mapping from case geometry keys such as
`d3_r3_z` to lists. Alphabetical group order resolves ties when choosing the
largest group as the reference.

Covariance initialization starts from the existing regularized estimator.
Uniform initialization starts from `uniform_probability`. The raw and
regularized covariance results remain available in either experiment design.
A failed covariance fit records its reason and supports either skipping the
covariance-initialized optimizer or using a separately labeled uniform fallback.

The realized initial priors are clipped to `[p_min, p_max]`, with separate
clipping counts. Let w₀ be their positive log odds. For group parameter θ, the
unnormalized edge weight is w₀ exp(θ) and the reference group uses θ = 0.
A monotone log-scale bisection finds the global factor that preserves the mean
of w₀ after clipping the resulting weights to the probability-induced bounds.
The realized mean may differ by at most 1e-11 from its target. Export reloads
the DEM and checks the probability vector, mechanism signatures, and realized
mean. If there are zero free parameters or the whole initial vector is at the
same bound, the workflow exports the initialization and records the status.

The diagonal Gaussian draws `population_size` candidates per generation.
Bounds clip the draws. The initial configuration and incoming incumbent are
scored on the current batch as extra controls. Identical θ vectors share one
evaluation, and stochastic draws alone contribute elite mass. The elite
cutoff splits tied mass evenly. The proposal smooths the first and second
moments and enforces a variance floor. Flat stochastic scores preserve both
proposal moments and the candidates remain eligible for checkpoint selection.
A deterministic subset without replacement is drawn per generation when
minibatching. The full prefix is used when `train_batch_shots` is null or
at least the prefix size. The best validation checkpoint minimizes failure
count, then squared θ norm, then generation index.

Positive global scaling preserves ideal MWPM decisions away from ties.
Fixing the mean weight sets a common scale within each optimization. A lower
failure probability measures decoder performance. Probability and
normalized-weight errors separately quantify recovery of the actual model.
Proposal variance describes the optimizer's search distribution.

## Decoders and reported failure rates

The tested versions are PyMatching 2.4.0, Stim 1.16.0, and ldpc 2.4.1. MWPM
constructs the matching graph from each candidate DEM and uses batch
predictions. UF uses direct `UnionFindDecoder` with singleton columns in H
and ranks cluster-growth candidates by LLR. With ldpc 2.4.1 on the tested
CPython 3.12 Linux wheel, omitting `uf_method` or passing a boolean raises
`TypeError`. The adapter supplies `uf_method=""` for peeling. The wrapper uses
a truthy string for matrix mode. The smoke fixture verifies that
`bits_per_step=1` changes the correction when LLR ordering reverses, that
`bits_per_step=0` grows all neighbors in that fixture, and that zero syndrome
has a zero correction. UF verifies the correction against H and computes
logical predictions with B. The evaluation reports an unsupported-prior
status if a baseline is outside a decoder's representable domain.
[ldpc wrapper source](https://github.com/quantumgizmos/ldpc/blob/d3429964cd4ffe1abfc041c6ec8b8425cb174f40/src_python/ldpc/union_find_decoder/_union_find_decoder.pyx)

Failure probability is the number of unsuccessful complete memory shots
divided by the total test shots. Each failed decoder correction contributes
one failure. Each successfully decoded shot with any wrong observable
contributes one logical mismatch. Conditional mismatch probability uses
the successful-decode count. Fixed-model intervals are 95% Wilson intervals.
On logarithmic plots, a triangle at the upper Wilson endpoint represents zero
observed failures. Paired comparisons give candidate-only failures n10,
baseline-only failures n01, and the difference (n10 − n01)/N.
The variation table reports spread in individual model rates separately from
fixed-model intervals. Shared test outcomes remain paired across trained
models and seeds.

## Artifacts and resources

| Location relative to the run output | Contents |
| --- | --- |
| `decoder_training.json`, `execution.json` | Scientific settings, information regime, versions, and execution resources |
| `models/<geometry>/<rate>/` | Actual canonical prior and DEM, compiled circuit, structure, group map, model provenance, and independent sample manifests |
| `fits/repNNN_nN/covariance.npz` under each case | Detector means and second moments, empirical detector covariance, raw signed pair probabilities and finite mask, and regularized mechanism priors when available |
| `fits/repNNN_nN/training/<initialization>/seedS/<decoder>/` | Initial probabilities, full group map, deterministic seeds and candidate history, checkpoints, selected θ, weights, probabilities, learned DEM, and content hashes |
| `evaluation.csv`, `training.csv`, `prior_diagnostics.csv` | Cross-decoder counts, paired comparisons, budgets, elapsed times, fit statuses, and error metrics |
| `evaluation_artifacts.json` and indexed NPZ files | Source detector IDs, prior matrices with a zero diagonal, singleton vectors and support masks, residuals against actual, and learned-versus-covariance discrepancies |
| `figures/` | PNG and PDF figures, the selected design, and separate replicate variation |

The error tables report full, pair, and singleton mechanism-vector RMSE and
MAE. Pair-matrix metrics count each unique detector pair once. Known-support
metrics use only canonical pairs, while all-pairs metrics include structural
zeros. Raw covariance pair metrics are finite-only, retain negative values,
and report coverage and invalid counts. An empty mask is unavailable. The
raw covariance-derived probability matrix and empirical detector covariance
are stored as separate arrays. Singletons have a separate vector and mask.
Group diagnostics include signed mean prior error and clipping information.
Sweep points have small horizontal display offsets by method so coincident
curves can be seen.

`resources.total_workers` cannot exceed the CPU allocation reported by the
package. Training requires `case_jobs × candidate_jobs × native_threads`
slots. Sampling and covariance fitting run in separate phases and use their
own worker counts. Candidate workers persist across generations, memory-map
the sample arrays read-only, and keep fixed decoder structure. Changing the
worker allocation or execution chunk sizes preserves the scheduled
candidates, batches, and results within the recorded dependency versions.
