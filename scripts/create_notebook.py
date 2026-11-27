"""Rebuild the single walkthrough notebook from reviewed source cells."""
from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook.cells = [
        md(r"""# THeatA LIZ predictive trigger prior calibration

This notebook uses the rotated surface-code workload of THeatA Figure 7. It
compiles each threshold, reconstructs the noisy Stim circuit and estimates
priors from calibration detections. It reports full-detector MWPM for
comparison with the paper and paired Z-sector MWPM for calibration.

The inferred parameters are the **identifiable edge priors of the memory-sector
marginal DEM**. The full circuit DEM may contain cross-sector hyperedges.
Identifying their separate priors from first and second moments depends on
the known support matrix; that general rank analysis is documented separately.
This implementation validates the exact projection and known logical labels.

See `docs/method.md` for the experimental protocol.
[THeatA §V-C and §V-D](https://arxiv.org/html/2611.30676v1#S5) ·
[Chen et al. (2021), Supplement VI.1](https://arxiv.org/html/2102.06132v1)
"""),
        md("""## 1. Configuration

Use the THeatA prior calibration kernel. The default settings use
`configs/lpt_calibration.yaml` and write to `results/lpt_calibration`. Choose
`configs/lpt_calibration_smoke.yaml` for a short run. Edit the overrides below
to customize the YAML settings. Set `RUN_EXPERIMENT = False` to review a saved
run with matching scientific settings.
"""),
        code(r"""from dataclasses import replace
import json
import os
from pathlib import Path
import sys

candidates = [Path(os.environ.get('THEATA_PRIOR_PROJECT', '.')).resolve(), Path.cwd().resolve(), Path.cwd().resolve().parent]
ROOT = next((p for p in candidates if (p / 'pyproject.toml').exists() and (p / 'src/theata_prior').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook within the theata-prior-calibration project.')
sys.path.insert(0, str(ROOT / 'src'))

from IPython.display import Markdown, display
from theata_prior.config import load_config
from theata_prior.experiment import build_models, cases, fit_directory, initialize_run, threshold_label
from theata_prior.plotting import load_metrics, plot_results, prior_rmse
from theata_prior.io import read_json

CONFIG = ROOT / 'configs' / os.environ.get('THEATA_PRIOR_CONFIG', 'lpt_calibration.yaml')
OUT = Path(os.environ.get('THEATA_PRIOR_OUT', ROOT / 'results' / CONFIG.stem)).expanduser().resolve()
RUN_EXPERIMENT = True

DISTANCES = None              # e.g. [3, 5, 7]
ROUNDS = None                 # e.g. 3, for every distance
LIZ_PT_THRESHOLDS = None      # e.g. [0.2, 0.5, 0.7, 1.0, 1.5, 2.1, 2.8]
LIZ_PT_MAX_ITERS = None       # e.g. 32
COOLING_DURATION_US = None    # e.g. 1000.0
RESIDUAL_PHONON = None        # e.g. 0.0 for all three modes
CALIBRATION_SHOTS = None      # e.g. [1_000, 10_000, 100_000]
LER_COMPARISON_SHOTS = None   # e.g. 10_000. None uses the largest calibration size for the figure.
TEST_SHOTS = None              # e.g. 10_000
REPETITIONS = None             # e.g. 2
WORKERS = None                 # e.g. 2
SEED = None                    # e.g. 20261127

overrides = dict(distances=DISTANCES, rounds=ROUNDS, liz_pt_thresholds=LIZ_PT_THRESHOLDS,
                 liz_pt_max_iters=LIZ_PT_MAX_ITERS, cooling_duration_us=COOLING_DURATION_US,
                 residual_phonon=RESIDUAL_PHONON, training_shots=CALIBRATION_SHOTS,
                 test_shots=TEST_SHOTS, repetitions=REPETITIONS, seed=SEED)
config = replace(load_config(CONFIG, workers=WORKERS),
                 **{key: value for key, value in overrides.items() if value is not None})
if config.liz_pt_thresholds is None:
    raise ValueError('Choose an LPT calibration configuration for this walkthrough.')
REPORT = OUT / 'walkthrough'
if not RUN_EXPERIMENT:
    if not (OUT / 'run.json').is_file():
        raise FileNotFoundError(f'No saved run at {OUT}')
    initialize_run(config, OUT)
print(json.dumps(config.to_dict(), indent=2))
print('Output:', OUT)
"""),
        md("""## 2. Inspect the compiled circuit and known structure

THeatA uses one reused syndrome ancilla, two resident coolant ions in the
single LIZ, `greedy_alt` placement, alternating forward and reverse rounds,
and the Mainz heating profile. It carries ion order across rounds and resets
the motional accumulators at each boundary. LPT temporarily parks the
data–ancilla pair in the adjacent Temp zone while cooling the resident ions.

The reconstructed CXs use THeatA Approach B with all 15 LUT probabilities and
per-ion lockstep idle dephasing. The trace records above-range lookups and
sector projection failures. Scheduled wall time and summed cooling time are
reported separately.
"""),
        code(r"""case_list = build_models(config, OUT) if RUN_EXPERIMENT else list(cases(config, OUT))
physical_rows = read_json(OUT / 'physical_metrics.json')

def show_value(value):
    return f'{value:.6g}' if value is not None else '—'

header = ('| d | Rounds | LPT threshold | Mean MS error | Cool ops | LPT cycles | Sum of cool ops (µs) | '
          'Scheduled time (µs) | Mean MS phonons | Max MS phonons | LUT out of range | Status |\n'
          '|---|---|---|---|---|---|---|---|---|---|---|---|\n')
body = '\n'.join(
    f"| {r['distance']} | {r['rounds']} | {threshold_label(r['cool_threshold'])} | "
    f"{show_value(r['mean_ms_error'])} | {r['cooling_operations']} | {r['lpt_cooling_cycles']} | {r['cooling_duration_us']:,.0f} | "
    f"{r['duration_us']:,.0f} | {r['mean_ms_phonon']:.6g} | {r['max_ms_phonon']:.6g} | "
    f"{r['lut_out_of_range_gates']} / {r['ms_gates']} | {r.get('projection_error', r['status'])} |"
    for r in physical_rows
)
display(Markdown(header + body))

case = next((c for c in reversed(case_list) if (c.directory / 'model.json').is_file()), None)
if case is not None:
    print(case.label)
    print('Compiled noisy circuit, first instructions:')
    print('\n'.join((case.directory / 'actual_noisy.stim').read_text().splitlines()[:18]))
"""),
        md("""## 3. Inspect calibration and independent heldout samples

Samples are drawn from the physical noisy Stim circuit. The full and Z-sector
evaluations use the same physical heldout shots. The covariance fit uses
detector samples from a separate calibration stream. Its independent DEM is
an approximation to the physical Pauli-channel distribution.

Each training size N takes the first N rows of a single maximum-size training
dataset. Test data use a different random namespace and are shared by all
compared decoders. Worker count can change without changing samples for fixed
shard size, source and Stim build.
"""),
        code(r"""if RUN_EXPERIMENT:
    from theata_prior.experiment import sample_models
    sample_models(config, OUT)

from theata_prior.sampling import SampleSet
if case is not None and (case.directory / 'known_structure.json').is_file():
    train = SampleSet.open(case.directory / 'samples_train_000', validate_observables=False)
    test = SampleSet.open(case.directory / 'samples_test')
    print('Largest calibration sample:', (train.shots, train.num_detectors))
    print('Heldout sample:', (test.shots, test.num_detectors))
    assert train.metadata['config']['stream'] != test.metadata['config']['stream']
"""),
        md(r"""## 4. Inspect the mechanism prior estimates

For $\mu_i=\langle x_i\rangle$ and $\mu_{ij}=\langle x_i x_j\rangle$, the exact
pair inversion is

$$\widehat p_{ij}=\frac12\left(1-\sqrt{
\frac{(1-2\mu_i)(1-2\mu_j)}{1-2\mu_i-2\mu_j+4\mu_{ij}}}\right).$$

Let $e$ index known mechanisms, $q_e$ be their priors, and $H_{ie}=1$ mean
that mechanism $e$ flips detector $i$. With $a_e=-\log(1-2q_e)$, define
$\widetilde P_{ij}=p_{ij}$ off the diagonal and $\widetilde P_{ii}=\mu_i$.
The entrywise transformation

$$\Gamma_{ij}=-\log(1-2\widetilde P_{ij})$$

gives the unified real-valued equation

$$\Gamma=H\operatorname{diag}(a)H^\top
=\sum_e a_e h_eh_e^\top.$$

In the implemented graphlike model, a mechanism with support $S_e=\{i,j\}$
has $q_e=p_{ij}$. For $S_e=\{i\}$, the diagonal equation isolates its prior:

$$q_e=\frac12\left[
1-\frac{1-2\mu_i}
{\prod_{\substack{f\ne e\\H_{if}=1}}(1-2q_f)}
\right].$$

This is S14--S16 expressed using the general mechanism index $e$.
See the [general derivation](../docs/general_prior_derivation.md) for the
step-by-step proof and the rank condition for arbitrary known supports.

The implementation uses stable equivalent forms. The full raw $p_{ij}$ matrix
retains negative finite-sample estimates; only known decoder edges are clipped
to the configured fixed interval. Boundary estimates use the regularized
incident pair priors. Diagnostics record all interventions.

S11 requires empirical detector means below 1/2. The notebook records fits
outside this range as undefined and continues with the available fits.
"""),
        code(r"""if RUN_EXPERIMENT:
    from theata_prior.experiment import fit_models
    fit_models(config, OUT, skip_saturated_marginals=True)

import numpy as np
from theata_prior.dem import GraphStructure

fit_dir = fit_directory(case, rep=0, shots=max(config.training_shots)) if case is not None else None
if fit_dir is not None and (fit_dir / 'fit_complete.json').is_file():
    p_ij = np.load(fit_dir / 'p_ij.npy', allow_pickle=False)
    print('p_ij matrix shape:', p_ij.shape)
    print('Diagnostics:')
    print(json.dumps(read_json(fit_dir / 'diagnostics.json'), indent=2)[:7000])
    print('Estimated DEM, first edges:')
    print('\n'.join((fit_dir / 'estimated.dem').read_text().splitlines()[:10]))
else:
    print('The selected fit is unavailable. See its status in the RMSE table.')
"""),
        md("""## 5. Paired MWPM evaluation

Actual and estimated sector decoders share the graph, logical labels and
heldout shots. The separate full-detector MWPM row matches the paper's
evaluation protocol. Each LER covers the complete memory experiment.

At threshold 0.2, Table III reports three-round full-detector LERs of
0.0039, 0.00097 and 0.00018 for distances 3, 5 and 7. The smoke preset
checks execution with a small sample.
[THeatA Table III](https://arxiv.org/html/2611.30676v1#S5.SS3)
"""),
        code(r"""if RUN_EXPERIMENT:
    from theata_prior.experiment import evaluate_models_stage
    evaluate_models_stage(config, OUT, skip_saturated_marginals=True)
rows = read_json(OUT / 'full_detector_metrics.json') + load_metrics(OUT)

header = '| d | LPT threshold | Calibration N | Decoder scope | Priors | Failures / test shots | LER | 95% Wilson interval |\n|---|---|---|---|---|---|---|---|\n'
body = '\n'.join(f"| {r['distance']} | {threshold_label(r['liz_pt_threshold'])} | {r['calibration_shots'] or '—'} | {r['decoder_scope']} | {r['decoder']} | {r['failures']} / {r['shots']} | {r['logical_error_rate']:.6g} | [{r['ci_low']:.6g}, {r['ci_high']:.6g}] |" for r in rows)
display(Markdown(header + body))
skips_path = OUT / 'fit_skips.json'
for fit in read_json(skips_path) if skips_path.is_file() else []:
    display(Markdown(
        f"**Undefined S11 fit** for d={fit['distance']}, threshold={threshold_label(fit['cool_threshold'])}, "
        f"N={fit['calibration_shots']:,}, repetition {fit['repetition']}. "
        f"{len(fit['saturated_detector_ids'])} detector means reached or exceeded 1/2."
    ))
"""),
        md(r"""## 6. Cooling policy, calibration size, and physical trade-offs

LER figures use the probability of failure over the complete experiment,
which is three rounds in the paper preset. Open triangles mark upper
confidence bounds for zero observed failures. The suppression ratio uses
$p_L(d)/p_L(d+2)$ at the same threshold and decoder setting. The ratio is
undefined for a zero failure count at distance $d+2$.

`LER_COMPARISON_SHOTS` selects the estimated series in the distance and
suppression-ratio figures. The default uses the largest calibration size.
Other figures show the physical metrics, prior RMSE, and the dependence of
LER and RMSE on the calibration size. Figures are exported as PNG and PDF.
"""),
        code(r"""comparison_shots = max(config.training_shots) if LER_COMPARISON_SHOTS is None else LER_COMPARISON_SHOTS
figure_paths = plot_results(OUT, split_ler=True, figure_dir=REPORT / 'figures',
                            comparison_shots=comparison_shots, include_no_cooling=False)
if not read_json(OUT / 'suppression_ratios.json'):
    print('The suppression ratio requires two evaluated distances separated by two.')
from IPython.display import Image
for path in figure_paths:
    if path.suffix == '.png' and not path.name.startswith('priors_'):
        display(Image(filename=str(path)))
"""),
        md(r"""## 7. Compare actual and estimated probabilities with RMSE

The raw S11 $p_{ij}$ comparison counts each unique off-diagonal pair once
($i<j$). It retains negative finite estimates and reports coverage for all
pairs and separately for known pair mechanisms. Actual probabilities are
zero at nonedges. Prior RMSE compares the saved regularized decoder priors
against the actual priors in canonical edge order, including both pair and
singleton mechanisms. Each RMSE is the square root of the mean squared
probability difference over the indicated entries.
"""),
        code(r"""import csv

metric_keys = ('raw_pair_rmse_all', 'raw_pair_rmse_support', 'prior_rmse',
               'finite_pairs', 'total_pairs', 'finite_known_pairs', 'total_known_pairs', 'mechanisms')
rmse_rows = []
for sweep_case in case_list:
    model = read_json(sweep_case.directory / 'model.json')
    built = (sweep_case.directory / 'known_structure.json').is_file()
    if built:
        structure = GraphStructure.from_dict(read_json(sweep_case.directory / 'known_structure.json'))
        actual = np.load(sweep_case.directory / 'actual_priors.npy', allow_pickle=False)
    for rep in range(config.repetitions):
        for count in config.training_shots:
            folder = fit_directory(sweep_case, rep=rep, shots=count)
            row = dict(distance=sweep_case.distance, rounds=sweep_case.rounds, liz_pt_threshold=sweep_case.cool_threshold,
                       calibration_shots=count, repetition=rep)
            if built and (folder / 'fit_complete.json').is_file():
                estimated = np.load(folder / 'estimated_priors.npy', allow_pickle=False)
                raw_pairs = np.load(folder / 'p_ij.npy', allow_pickle=False)
                row.update(status='ok', **prior_rmse(structure, actual, estimated, raw_pairs))
            else:
                status = 'undefined' if (folder / 'fit_skipped.json').is_file() else 'missing'
                row.update(status=status if built else model['projection_error'], **dict.fromkeys(metric_keys))
            rmse_rows.append(row)

report_csv = REPORT / 'prior_rmse.csv'
REPORT.mkdir(parents=True, exist_ok=True)
with report_csv.open('w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rmse_rows[0]))
    writer.writeheader()
    writer.writerows(rmse_rows)

def show_rmse(value):
    return f'{value:.6g}' if value is not None else '—'

header = ('| d | LPT threshold | Calibration N | Repetition | Raw p_ij RMSE, all pairs | Finite / all | '
          'Raw p_ij RMSE, known pairs | Finite / known | Prior RMSE | Status |\n'
          '|---|---|---|---|---|---|---|---|---|---|\n')
body = []
for row in rmse_rows:
    all_pairs = (f"{row['finite_pairs']:,} / {row['total_pairs']:,}" if row['status'] == 'ok' else '—')
    known_pairs = (f"{row['finite_known_pairs']:,} / {row['total_known_pairs']:,}" if row['status'] == 'ok' else '—')
    body.append(
        f"| {row['distance']} | {threshold_label(row['liz_pt_threshold'])} | {row['calibration_shots']:,} | {row['repetition']} "
        f"| {show_rmse(row['raw_pair_rmse_all'])} | {all_pairs} | {show_rmse(row['raw_pair_rmse_support'])} "
        f"| {known_pairs} | {show_rmse(row['prior_rmse'])} | {row['status']} |")
display(Markdown(header + '\n'.join(body)))
"""),
        md("""## 8. Inspect recovered priors and saved artifacts

The first two panels compare the actual pair-probability matrix and raw S11
matrix. The third compares pair and boundary priors. Negative raw estimates
remain visible in the raw matrix; floor-clipped decoder probabilities appear
on the floor line in the scatter plot. True priors are accessed only for this
post-fit comparison.
"""),
        code(r"""for prior_figure in figure_paths:
    if prior_figure.suffix == '.png' and prior_figure.name.startswith('priors_'):
        display(Image(filename=str(prior_figure)))
if fit_dir is not None and (fit_dir / 'prior_error.json').is_file():
    print('Prior error diagnostics:', read_json(fit_dir / 'prior_error.json'))
print('Loaded experiment:', OUT)
print('RMSE table:', report_csv)
print('Standalone figures:', REPORT / 'figures')
"""),
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "THeatA prior calibration", "language": "python", "name": "theata-prior"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    target = Path(__file__).resolve().parents[1] / "notebooks/calibration_walkthrough.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
