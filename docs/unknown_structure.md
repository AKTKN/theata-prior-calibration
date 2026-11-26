# Unknown detector-support benchmark

The benchmark learns the canonical detector supports and priors of the same
selected-sector graphlike marginal used by the known-support experiment.
It considers every singleton and every unordered pair of observed detectors,
including isolated detectors and nonlocal pairs. The candidate count is
\(M(M+1)/2\). Array order is lexicographic pairs followed by singletons in
local detector order. Original detector IDs and coordinates are obtained from
the noiseless Stim circuit. Coordinates are available for interpretation and
plots, while tests and fitting use the same complete family for any geometry.

The model has independent Bernoulli mechanisms of detector weight one or two
and rates below one half. Mechanisms with an identical detector signature
have an identifiable XOR-aggregate rate. Recovery concerns these aggregate
supports rather than microscopic faults. THeatA compiles the standard
rotated-memory circuit. The simulator retains the full physical DEM and
validates its exact marginal for the declared memory basis. Each Stim
error instruction remains one event when it includes separators. Residual
higher-weight mechanisms and inconsistent logical actions cause a build
error. The primary source is independent DEM sampling. Setting
sample_source to circuit selects a labeled model-mismatch experiment.
Physical p is the mean total nonidentity Pauli error probability at
physical MS locations.

## Run the experiment

Run these commands from the project root. Choose a fresh output directory
when changing a scientific setting. A run can resume in its original directory.

~~~bash
bash scripts/bootstrap.sh
bash scripts/run_structure.sh --config configs/structure_smoke.yaml --out results/structure_smoke
bash scripts/run_structure.sh --config configs/structure_demo.yaml --out results/structure_demo
.venv/bin/theata-prior structure preflight --config configs/structure_production.yaml
bash scripts/run_structure.sh --config configs/structure_production.yaml --out results/structure_production
.venv/bin/jupyter lab notebooks/structure_learning_walkthrough.ipynb
~~~

Choose the THeatA prior calibration kernel. The
[walkthrough](../notebooks/structure_learning_walkthrough.ipynb) runs the
detector-only fit first, inspects its marginal DEM, then attaches optional
calibration-observable labels and scores the results.

The CLI namespace supports build, sample, fit, label, evaluate, plot, run, and
preflight. For example, use the following commands to run the blind stage
against archived samples and then inspect the final figures.

~~~bash
.venv/bin/theata-prior structure fit --config configs/structure_smoke.yaml --out results/structure_smoke
.venv/bin/theata-prior structure plot --out results/structure_smoke
~~~

The Python entry points are `load_structure_config` and
`run_structure_experiment` for a whole sweep, `fit_structure` for
calibration detector arrays, `invert_detector_moments` for population
or empirical moment inversion, and `fit_logical_labels` for augmented
calibration parities. The walkthrough calls the same stage APIs as the CLI.

The strict configuration consists of the base experiment options plus a
structure mapping. Set label_mode to detector_only for a standalone
detector-recovery run. The supplied configurations opt in to
calibration_observables. The selection settings are alpha_pairs and
alpha_singletons. The shared probability_ceiling caps the fitted rates below
one half. The base probability_floor and strict_estimation apply to the
known-support comparator. The resource-only case_jobs option sets the maximum
number of simultaneous case fits or label jobs. The worker budget is shared
among jobs and capped by the available allocation. Build and sampling use
the existing simulation APIs.

## Exact detector inversion

For Boolean outcomes \(D_i\), define \(b_i=1-2E[D_i]\) and
\(b_{ij}=1-2E[D_i]-2E[D_j]+4E[D_iD_j]\). A parity bias is the product
of \(1-2q_S\) over supports that flip it. With
\(\lambda_i=-\log b_i\), \(\lambda_{ij}=-\log b_{ij}\), and
\(a_S=-\log(1-2q_S)\), the equations give

\[
a_{ij}=(\lambda_i+\lambda_j-\lambda_{ij})/2,\qquad
a_i=\lambda_i-\sum_{j\ne i} a_{ij}.
\]

The pair probability is exactly S11,
\(q_{ij}=(1-\sqrt{b_i b_j/b_{ij}})/2\), and
\(q_i=-\operatorname{expm1}(-a_i)/2\). With pairs ordered first, the
attenuation system is block triangular with identity diagonal blocks and
the unsigned node-pair incidence matrix in the lower left. The population
map is invertible even with absent supports. S11 is evaluated using the
stable pair-probability implementation from the known-support experiment.
The S11 plot diagonal uses zero. Singleton residuals start from the
measured detector means. The archive preserves signed raw rates and records
negative attenuations and undefined logarithms. A nonpositive marginal or
pair parity leaves the complete fit unresolved, and the oracle baseline
can still be evaluated.

## Finite-shot comparisons

- **actual_prior** uses oracle support, priors, and labels during evaluation.
- **known_support_estimated** uses the existing estimator with oracle
  support and labels and the same detector-only calibration prefix.
- **complete_clipped** applies zero clipping and the upper cap to every raw
  pair. It recomputes singleton residuals with those pairs and applies the
  same bounds. Its reported selection mask is the final positive support.
- **complete_selected** screens the entire candidate family, sets rejected
  pairs to zero, and refits singleton residuals using the retained pairs.
  Only selected singletons are eligible for positive rates. Initial
  decisions, refit attenuations, and final active masks are archived.
  The active DEM contains positive fitted rates. Under the default policy,
  exact zeros remain absent.

Under the independent-event graphlike model, a missing pair mechanism makes
its two detectors independent, since the two sets of underlying events are
disjoint even if the graph contains an indirect path. Integer coincidence
counts determine the one-sided Fisher exact tail
\(\Pr(X\ge n_{11})\) for \(X\sim\mathrm{Hypergeom}(N,n_i,n_j)\). Holm
correction at alpha_pairs (default 0.025) covers the full predetermined
pair family including degenerate tables, whose p-value is one. This controls
pair family-wise Type I error at the nominal level under the stated model
for a fixed prefix and case, without a between-tests independence
assumption. [SciPy Fisher test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html) · [Holm procedure](https://www.jstor.org/stable/4615733)

The singleton Wald test uses the **unclipped complete** residual, including
pairs later rejected. Write \(z_i=(-1)^{D_i}\) and \(h_i=(z_i-b_i)/b_i\).
With \(W_{ij}=1/b_{ij}\) off the diagonal and zero on the diagonal, one
shot's influence on \(a_i\) is

\[
\mathrm{IF}_{a_i}
=\frac{M-4}{2}h_i+\frac12\sum_j h_j
 +\frac{M-1}{2}-\frac12 z_i(zW)_i.
\]

The standard error is the sample standard deviation of the influence divided
by \(\sqrt{N}\). Shot chunks and detector row blocks bound memory.
Holm correction at alpha_singletons (default 0.025) covers all detectors.
A zero estimated standard error or an all-zero observed detector column
gives p-value one under this screen. This rule expresses insufficient
evidence for selection. Singleton inference is asymptotic and can be
unreliable for rare or near-saturated parities. The benchmark gives no joint
finite-shot 0.05 guarantee over singles, pairs, sweep points, and prefixes.
Selection and rate reconstruction use the same calibration prefix, which
can introduce selection bias. Neither the heldout samples nor a hidden
prior, edge count, or degree influences the selection policy.

## Logical labels and decoding

Using \(D=HF \bmod 2\) and \(L=BF \bmod 2\), detector data determine the
model's detector distribution independently of \(B\). Detector-only mode
therefore scores learned supports and priors and emits logical-free marginal
DEMs. LER is available for the actual and known-support references in this
mode. The two learned methods have unavailable LER entries.

The optional calibration-observable stage supports one logical bit. For
a subset \(U\) of detector and observable indices, let \(\lambda_U\)
be minus the log of its empirical parity bias. The raw label-1 pair
attenuation and corresponding singleton attenuation are

\[
T_{ij}=(\lambda_i+\lambda_j+\lambda_L-\lambda_{ij}
        -\lambda_{iL}-\lambda_{jL}+\lambda_{ijL})/4,\qquad
a_{i,1}=(\lambda_i+\lambda_L-\lambda_{iL})/2-\sum_{j\ne i}T_{ij}.
\]

The label-0 attenuation is the complete-family detector attenuation minus
the label-1 attenuation. These equations follow by taking intersections
of the contributing parity event sets. The remaining pure-logical residual
is \(\lambda_L-\sum_i a_{i,1}-\sum_{i<j}T_{ij}\). Signed coincidence counts
compute augmented parities in bounded shot blocks. A positive
label-1 minus label-0 contrast assigns bit one, and other contrasts
assign zero. Exact zero ties are marked ambiguous. Variant estimates,
contrasts, the magnitude of the smaller absolute variant attenuation,
and the pure-logical residual are available as consistency diagnostics.
Empirical residuals can come from sampling variation. The simulator
restricts the reference model to one logical label per support and rejects
detector-invisible logical events. Nonpositive required logical parities
skip supervised labeling for that fit.

Labeling freezes the detector fit rates and support masks. It uses the
same calibration D and L0 prefix. Test data enter exclusively at scoring.
Every decoder receives the same heldout shots. An odd syndrome in a
component without a boundary is a decoding failure. The main LER counts
either a decoding failure or a wrong prediction over all heldout shots,
and the conditional rate uses successfully decoded shots only. There are
no fallback oracle edges. Paired intervals use discordant counts.
Wilson intervals describe a fixed fitted decoder. The figures show means
and an envelope of per-fit Wilson intervals across calibration repeats.
They avoid pooling reuses of a heldout shot as independent trials.

## Artifacts and cost

- **structure_run.json** defines the experiment, model class, support and
  label policies, and package versions. **simulation/** holds the existing
  base build and sample caches in a separate namespace. The retained
  samples are reproducible nested prefixes with independent train/test
  streams. Resumption checks scientific settings, model and sample identity,
  and hashed completed fit artifacts. BLAS worker or block changes do not
  change scientific identity. Counts and masks matched in the worker-count
  test. Singleton standard errors agreed within \(10^{-12}\) relative
  tolerance on that fixture. Floating-point comparisons near a test
  threshold can be sensitive to numerical reductions.
- **blind/** contains detector_metadata.json, detector_fit.npz, fit.json,
  detector-marginal DEMs, and the optional logical-label arrays and DEMs.
  Moment counts, probabilities, raw and refitted attenuation, test
  statistics, adjusted p-values and masks remain in complete-family order.
- **evaluation/** joins canonical sorted support keys and records truth,
  per-detector spurious incident mass, and evaluation manifests. Summary
  CSVs report support, prior, label, decoding, and paired errors.
  A zero metric denominator is blank in CSV and null in JSON. Invalid fits
  are listed in skipped_conditions.json and count as unresolved supports.
  Support/prior metrics use all positive oracle rates as truth, including
  small rates. Heldout moment prediction errors compare the fitted model
  against independent empirical moments after fitting has finished.
- **figures/** contains LER curves, support and prior sweep summaries,
  a support matrix comparison, raw estimate histograms, and optional
  label figures. Support/prior/label figures show the smallest distance
  and highest physical p. The matrix and histograms use its largest
  calibration prefix, repetition zero. **timings.json** and
  **preflight.json** record measured stage durations, process RSS
  high-water marks when available, and estimated array sizes.

Dense statistics need roughly \(O(NM^2)\) arithmetic and
\(O(M^2+B M)\) numerical memory for a shot block of size \(B\). Signed
label counts and influence-function sweeps have similar costs. The
number of pair tests scales as \(O(d^6)\) for rounds proportional to d.
Prefix moments and signed counts reuse cumulative integer counts.
Memory-mapped datasets need approximately
\((M+1)(N_{\max} \times \mathrm{repetitions}+N_\mathrm{test})\) bytes per
physical p. The preflight working-array estimate is per concurrent fit
and excludes THeatA, Python library state, disk-backed sample caches,
and matching. The production configuration has 80,200 candidates and
1.604 GB of sample files per physical p at d=9. Its d=9 maximum-prefix
moment pass uses roughly 160 billion multiply-accumulates.

## Literature context

All-pair covariance inversion and significance pruning have prior
precedent. The contribution here is a reproducible THeatA hardware-model
benchmark for complete-family graphlike support recovery, calibration-shot
cost, and decoding with declared label information.
[Spitz et al., Section IV.1](https://arxiv.org/abs/1712.02360) ·
[Chen et al., Supplement VI](https://arxiv.org/html/2102.06132v1) ·
[Blume-Kohout and Young, Section 4](https://arxiv.org/html/2504.14643v1)

Methods for pruning and related empirical reconstruction work put the
comparison in context. Physically motivated signature families and
logical-outcome information are additional sources of structure.
[Arms et al.](https://arxiv.org/html/2512.10814v2) ·
[Ye and Zhang](https://cpb.iphy.ac.cn/en/article/doi/10.1088/1674-1056/ae663a) ·
[Remm et al.](https://arxiv.org/html/2502.17722v1) ·
[Takou and Brown](https://arxiv.org/html/2504.20212v1)

## Executed smoke result

The shell command above completed in about 8 seconds on this machine.
It used Python 3.12.7, Stim 1.16.0, PyMatching 2.4.0, NumPy 2.5.5,
SciPy 1.18.1, Matplotlib 3.11.2, and THeatA at
8bfb9b97de91bacac2f56725c27b4c71c6a91009. The dedicated Jupyter
kernel executed all five code cells from a separate clean output directory
and produced byte-identical metric tables. A two-rate detector-only smoke
run also completed with two case jobs sharing two workers. The production
configuration was sized with preflight. The demo and production sweeps
were not launched.

The test suite used the following command from the project root.

~~~bash
THEATA_PATH="$PWD/vendor/theata" .venv/bin/python -m pytest -q
~~~

All 54 tests passed. New checks include independent Bernoulli enumeration,
finite differences and Monte Carlo for singleton uncertainty, count and
Holm orientation checks, data-boundary and resume tests, and a deliberately
unmatchable decoder graph.

At d=3 and physical p=0.002, there were 16 detectors and 55 true supports
among 136 candidates. Independent heldout size was 10,000 shots.
Conditional mismatch equals the end-to-end rate here because each of
the evaluated smoke models had zero decoding failures. A separate
pruned-graph test exercises the decoding-failure denominator.

| Method | Calibration N | TP / FP / FN | True-support prior RMSE | Label errors on matched supports | End-to-end LER (95% Wilson) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual priors and labels | Baseline | 55 / 0 / 0 | 0 | Known | 0.0161 [0.0138, 0.0188] |
| Known support and labels | 1,000 | 55 / 0 / 0 | 0.003933 | Known | 0.0232 [0.0204, 0.0263] |
| Complete clipped, calibration-observable labels | 1,000 | 42 / 42 / 13 | 0.006589 | 2 / 42 | 0.0569 [0.0525, 0.0616] |
| Complete selected, calibration-observable labels | 1,000 | 22 / 0 / 33 | 0.010211 | 0 / 22 | 0.1085 [0.1026, 0.1147] |
| Known support and labels | 10,000 | 55 / 0 / 0 | 0.001347 | Known | 0.0155 [0.0133, 0.0181] |
| Complete clipped, calibration-observable labels | 10,000 | 55 / 42 / 0 | 0.001598 | 0 / 55 | 0.0173 [0.0149, 0.0200] |
| Complete selected, calibration-observable labels | 10,000 | 47 / 0 / 8 | 0.002295 | 0 / 47 | 0.0332 [0.0299, 0.0369] |

There were zero ambiguous labels on matched true supports in this smoke
run. The clipped model's false-positive prior mass was 0.0760 at 1,000
shots and 0.0147 at 10,000 shots. The selected model's mass was zero at
both sizes, with recall 0.400 and 0.855. This single repetition shows
how support precision and decoding performance can differ.
