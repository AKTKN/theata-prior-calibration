"""Rebuild the single walkthrough notebook from reviewed source cells."""
from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook.cells = [
        md(r"""# THeatA surface-code prior calibration

This notebook runs the complete experiment through the same modules as the CLI:
standard Stim circuit → THeatA native compilation → actual noisy model →
independent calibration/test samples → exact covariance and boundary inversion
→ paired MWPM evaluation → two LER figures.

The inferred parameters are the **identifiable edge priors of the memory-sector
marginal DEM**. The full circuit DEM may contain cross-sector hyperedges.
Identifying their separate priors from first and second moments depends on
the known support matrix; that general rank analysis is documented separately.
This implementation validates the exact projection and known logical labels.

Reference: [Chen et al. (2021), Supplement VI.1, S11 and S14–S16](https://arxiv.org/html/2102.06132v1).
See `docs/method.md` for the complete assumptions and sampling definitions.
"""),
        md("""## 1. Configuration

Run `bash scripts/bootstrap.sh` first and select the THeatA prior calibration
kernel. `THEATA_PATH` can select an existing checkout. This walkthrough uses
`configs/main1.yaml` and saves results in `results/main1`. Set `CONFIG` to
`configs/smoke.yaml` for a short execution check. The output directory
follows the configuration filename.
The physical rate p is the mean total nonidentity MS-gate error probability.
"""),
        code(r"""import json
import os
from pathlib import Path
import sys

candidates = [Path(os.environ.get('THEATA_PRIOR_PROJECT', '.')).resolve(), Path.cwd().resolve(), Path.cwd().resolve().parent]
ROOT = next((p for p in candidates if (p / 'pyproject.toml').exists() and (p / 'src/theata_prior').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook within the extracted theata-prior-calibration project.')
sys.path.insert(0, str(ROOT / 'src'))

from theata_prior.config import load_config
from theata_prior.experiment import build_models, sample_models, fit_models, evaluate_models_stage, fit_directory
from theata_prior.plotting import plot_results
from theata_prior.io import read_json

CONFIG = ROOT / 'configs/main1.yaml'
OUT = ROOT / 'results' / CONFIG.stem
config = load_config(CONFIG)
print(json.dumps(config.to_dict(), indent=2))
"""),
        md("""## 2. Compile the standard circuit and construct the known structure

The compiler keeps all original Stim data and ancilla qubits. Exact native
Clifford identities and the upstream sequential/parallel hardware validators
are checked. The supplied YAML enables threshold cooling and rejects any MS
lookup beyond the tabulated phonon range.
"""),
        code(r"""case_list = build_models(config, OUT)
case = case_list[-1]
hardware = read_json(case.directory / 'hardware.json')
model = read_json(case.directory / 'model.json')
print(json.dumps({k: model[k] for k in ['distance', 'rounds', 'physical_error_rate', 'full_detectors', 'sector_detectors', 'sector_errors']}, indent=2))
print('Mean physical MS error:', hardware['mean_ms_error'])
print('THeatA duration [s]:', hardware['duration_us'] / 1e6)
print('Compiled noisy circuit, first instructions:')
print('\n'.join((case.directory / 'actual_noisy.stim').read_text().splitlines()[:18]))
"""),
        md("""## 3. Sample calibration and independent heldout data

The default source is the full actual **independent DEM**, with selected
detectors and all logical flips retained. `sample_source: circuit` instead
samples the compiled physical circuit and includes circuit-to-independent-DEM
model mismatch. Logical labels are never used for prior fitting.

Each training size N takes the first N rows of a single maximum-size training
dataset. Test data use a different random namespace and are shared by all
compared decoders. Worker count can change without changing samples for fixed
shard size, source and Stim build.
"""),
        code(r"""sample_models(config, OUT)
from theata_prior.sampling import SampleSet
train = SampleSet.open(case.directory / 'samples_train_000', validate_observables=False)
test = SampleSet.open(case.directory / 'samples_test')
print('Calibration detectors:', train.load_detectors().shape)
print('Heldout detectors:', test.load_detectors().shape)
print('Heldout logical flips:', test.load_observables().shape)
assert train.metadata['config']['stream'] != test.metadata['config']['stream']
"""),
        md(r"""## 4. Estimate mechanism priors

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
        code(r"""fit_models(config, OUT, skip_saturated_marginals=True)
import numpy as np
from theata_prior.dem import GraphStructure

fit_dir = fit_directory(case, rep=0, shots=max(config.training_shots))
structure = GraphStructure.from_dict(read_json(case.directory / 'known_structure.json'))
if (fit_dir / 'fit_complete.json').is_file():
    p_ij = np.load(fit_dir / 'p_ij.npy')
    fitted_priors = np.load(fit_dir / 'estimated_priors.npy')
    moments = np.load(fit_dir / 'moments.npz')
    print('p_ij matrix shape:', p_ij.shape)
    print('Diagnostics:')
    print(json.dumps(read_json(fit_dir / 'diagnostics.json'), indent=2)[:7000])
    print('Estimated DEM, first edges:')
    print('\n'.join((fit_dir / 'estimated.dem').read_text().splitlines()[:10]))
else:
    print('The selected fit is undefined. See fit_skips.json after evaluation.')
"""),
        md("""## 5. Paired MWPM evaluation

Both actual and inferred decoders use the same graph and logical edge labels.
Logical failure means at least one predicted observable differs from the
heldout label, per complete memory experiment. Reported Wilson intervals are
conditional on each calibration dataset. Paired differences use discordant
failure counts, not an independence assumption between decoders.
"""),
        code(r"""rows = evaluate_models_stage(config, OUT, skip_saturated_marginals=True)
from IPython.display import Markdown, display

header = '| d | p | Calibration N | Decoder | Failures / test shots | LER | 95% Wilson interval |\n|---|---|---|---|---|---|---|\n'
body = '\n'.join(f"| {r['distance']} | {r['physical_error_rate']:g} | {r['calibration_shots'] or 'actual'} | {r['decoder']} | {r['failures']} / {r['shots']} | {r['logical_error_rate']:.6g} | [{r['ci_low']:.6g}, {r['ci_high']:.6g}] |" for r in rows)
display(Markdown(header + body))
for fit in read_json(OUT / 'fit_skips.json'):
    display(Markdown(
        f"**Undefined S11 fit** for d={fit['distance']}, p={fit['physical_error_rate']:g}, "
        f"N={fit['calibration_shots']:,}, repetition {fit['repetition']}. "
        f"{len(fit['saturated_detector_ids'])} detector means reached or exceeded 1/2."
    ))
"""),
        md("""## 6. Physical-error and calibration-shot sweeps

Solid curves use fitted priors; dashed curves use actual priors. The second
figure varies **calibration shots**, keeping heldout evaluation size fixed.
On logarithmic plots, zero-error counts appear at their upper confidence bound
with open downward triangles. CSV files retain zero as the point estimate.
"""),
        code(r"""figure_paths = plot_results(OUT)
from IPython.display import Image
display(Image(filename=str(OUT / 'figures/ler_vs_physical_error.png')))
display(Image(filename=str(OUT / 'figures/ler_vs_calibration_shots.png')))
"""),
        md("""## 7. Inspect recovered priors and saved artifacts

The first two panels compare the actual pair-probability matrix and raw S11
matrix. The third compares pair and boundary priors. Negative correlations
remain visible in the raw matrix; floor-clipped decoder probabilities appear
on the floor line in the scatter plot. True priors are accessed only for this
post-fit comparison.
"""),
        code(r"""prior_figure = OUT / f'figures/priors_d{case.distance}.png'
if prior_figure in figure_paths:
    display(Image(filename=str(prior_figure)))
if (fit_dir / 'prior_error.json').is_file():
    print('Prior error diagnostics:', read_json(fit_dir / 'prior_error.json'))
print('Saved results:', OUT)
print('Figures:', [p.name for p in figure_paths])
print('CSV tables: metrics.csv, paired_comparisons.csv, per-fit prior_comparison.csv')
print('Undefined fits:', OUT / 'fit_skips.json')
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
