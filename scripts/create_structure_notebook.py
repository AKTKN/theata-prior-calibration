"""Build the executable unknown-support walkthrough from source cells."""
from pathlib import Path

import nbformat as nbf


def main() -> None:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook(cells=[
        md("""# Learning unknown graphlike detector supports

This walkthrough uses the independent THeatA DEM from a distance-3 Stim
surface-code memory experiment. It considers every detector singleton and
pair. The detector-only fit uses calibration detections. A separate optional
stage uses calibration logical outcomes to attach L0 labels to the fixed
detector fits. The evaluation stage compares support, priors, labels, and
decoding against the simulation reference with independent heldout shots.

Run the project setup script and select the **THeatA prior calibration**
kernel. The configuration uses two nested calibration prefixes and the same
10,000 heldout shots for each decoder.
"""),
        code("""from pathlib import Path
import json

import numpy as np
from IPython.display import Markdown, Image, display

from theata_prior.candidates import DetectorMetadata
from theata_prior.io import read_json
from theata_prior.structure_config import load_structure_config
from theata_prior import structure_experiment as experiment

cwd = Path.cwd().resolve()
root = next((path for path in (cwd, *cwd.parents)
             if (path / 'configs/structure_smoke.yaml').is_file()), None)
if root is None:
    raise RuntimeError('Open Jupyter from within the calibration project.')
config = load_structure_config(root / 'configs/structure_smoke.yaml')
out = root / 'results/structure_notebook'
print(json.dumps(experiment.preflight_structure(config), indent=2))
"""),
        md("""## Detector-only support and prior fit

Build and sample the independent DEM. The compiler and sampler archive their
simulation inputs. The fitter receives the sanitized detector map and the
calibration detector arrays. Fisher tests and Holm correction screen all
pair candidates. Complete-family singleton residuals and their influence
standard errors determine the singleton screen. The selected model refits
rates on that support.
"""),
        code("""experiment.build_structure(config, out)
experiment.sample_structure(config, out)
experiment.fit_structure_stage(config, out)

metadata_path = next((out / 'blind').glob('d*/p*/detector_metadata.json'))
metadata = DetectorMetadata.from_dict(read_json(metadata_path))
n = max(config.base.training_shots)
fit_dir = metadata_path.parent / f'rep000_n{n}'
fit_status = read_json(fit_dir / 'fit.json')
print('Fit status', fit_status['status'])
with np.load(fit_dir / 'detector_fit.npz') as fit:
    print('Raw S11 block, local detectors 0 through 3')
    print(np.round(fit['raw_p_ij'][:4, :4], 5))
    if 'complete_selected_active' in fit:
        active = np.flatnonzero(fit['complete_selected_active'])
        print('First ten final active supports and priors')
        print([(metadata.candidates.support(i), round(float(fit['complete_selected_priors'][i]), 6))
               for i in active[:10]])
        print('Detector-marginal DEM', (fit_dir / 'complete_selected_detector_only.dem').relative_to(out))
"""),
        md("""## Optional supervised logical labels

A detector-only DEM supports a recovery comparison of detector signatures and
priors. Logical decoding of the learned graph requires additional label
information. In calibration-observable mode the same calibration prefix
supplies L0 for a parity-contrast sign decision. Each detector prior and
active-support mask stays fixed through this stage. Repeat with
detector-only mode and a fresh output directory to omit this stage.
"""),
        code("""experiment.label_structure_stage(config, out)
label_path = fit_dir / 'labels.json'
if label_path.exists():
    labels = read_json(label_path)
    keys = ('status', 'policy', 'ambiguous_candidates', 'pure_logical_attenuation')
    print({key: labels.get(key) for key in keys})
"""),
        md("""## Support and logical decoding on heldout data

The actual-prior baseline uses actual support, priors, and labels. The
known-support comparison estimates priors from the same detector data with
the actual support and labels supplied. The two complete-family methods use
their supervised label estimates in this configuration. Unmatchable shots
count toward the end-to-end error rate. Wilson intervals describe a fixed
fitted decoder tested on IID heldout shots.
"""),
        code("""tables = experiment.evaluate_structure_stage(config, out)

header = ('| Method | Information | N | Errors / shots | End-to-end rate (95% Wilson) | Unmatched | '
          'Conditional mismatch |\\n|---|---|---:|---:|---:|---:|---:|')
lines = [header]
for row in tables['decoding']:
    n_display = f"{row['calibration_shots']:,}" if row['calibration_shots'] else 'Baseline'
    if row['status'] == 'complete':
        value = f"{row['logical_error_rate']:.5f} [{row['ci_low']:.5f}, {row['ci_high']:.5f}]"
        errors = f"{row['failures']:,} / {row['shots']:,}"
        unmatched = f"{row['decoding_failures']:,}"
        conditional = row['conditional_logical_mismatch_rate']
        conditional_display = f"{conditional:.5f}" if conditional is not None else 'Undefined'
    else:
        value, errors, unmatched, conditional_display = row['status'], '', '', ''
    lines.append(f"| {row['method']} | {row['information']} | {n_display} | {errors} | {value} | "
                 f"{unmatched} | {conditional_display} |")
display(Markdown('\\n'.join(lines)))
for row in tables['support']:
    if row['support_type'] == 'combined' and row['method'].startswith('complete_'):
        print(row['method'], row['calibration_shots'],
              {name: row[name] for name in ('tp', 'fp', 'fn', 'precision', 'recall')})
"""),
        md("""## Saved diagnostics and figures

The plots and CSV tables come from the package modules. The diagonal of the
support matrix displays singleton decisions. The raw S11 matrix uses a zero
diagonal for display. The end-to-end error rate includes decoder failures,
while the conditional rate uses successfully decoded shots.
"""),
        code("""figures = experiment.plot_structure_stage(out)
for name in ('support_matrix_comparison.png', 'ler_vs_calibration_shots.png'):
    display(Image(filename=str(out / 'figures' / name)))
print('Figures', [path.name for path in figures])
print('Output directory', out.relative_to(root))
"""),
    ])
    notebook.metadata = {
        "kernelspec": {"display_name": "THeatA prior calibration", "language": "python", "name": "theata-prior"},
        "language_info": {"name": "python"},
    }
    target = Path(__file__).resolve().parents[1] / "notebooks/structure_learning_walkthrough.ipynb"
    nbf.write(notebook, target)


if __name__ == "__main__":
    main()
