# Validation and demonstration results

## Cooling-threshold calibration

The [walkthrough](notebooks/calibration_walkthrough.ipynb) compiles each
distance, round count, basis, and cooling threshold independently. The
end-to-end Z-memory smoke run used d=3, three rounds, calibration prefixes
of 2,000 and 10,000 shots, and a shared 10,000-shot heldout set per condition.
The evaluation below uses the 10,000-shot fitted priors.

| Threshold | Mean MS error | Cool ops | Summed cooling (ms) | Wall time (ms) | Actual LER | Estimated LER | Prior RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.002433164 | 72 | 28.8 | 458.735 | 0.0276 | 0.0291 | 0.00141586 |
| 1.5 | 0.002569084 | 69 | 27.6 | 457.635 | 0.0257 | 0.0255 | 0.00148225 |
| 4 | 0.007417600 | 38 | 15.2 | 446.007 | 0.0400 | 0.0402 | 0.00220935 |
| No cooling | 0.109683889 | 0 | 0 | 431.311 | 0.4481 | 0.4688 | 0.122333 |

The smoke run enables the upstream LUT boundary behavior. The no-cooling trace
has 50 of 72 MS gates above the LUT maximum of 20 transverse phonons. The
2,000-shot fit in that condition is undefined because a detector mean reached
1/2. With strict lookup enabled, the same condition retains the phonon trace
and skips noise generation. At threshold 1.5 the reloaded noisy circuit and
both full DEMs match the legacy implementation at `alpha=1.0` exactly.

## Historical physical-error-rate sweep

The former calibration walkthrough compiled THeatA once per code distance
through `build_models()` and `prepare_surface()`.

1. The bridge expands standard Stim gates such as `H` and `CX` into THeatA
   operations and constructs the dependency graph and initial ion placement.
   [Stim bridge](src/theata_prior/circuits.py#L309)
2. `result = compiler.compile(dag, initial)` returns the hardware schedule.
   The in-memory `result.timesteps` contains parallel operations with
   transport and cooling. `result.flat_ops` contains the list before
   scheduling. [THeatA compile result](https://github.com/MSTakahashiPJ/theata/blob/8bfb9b97de91bacac2f56725c27b4c71c6a91009/src/qccd_compiler.py#L2618)
3. The simulator applies idle times and the MS-gate phonon trace to
   construct `actual_noisy.stim`. The notebook displays its first 18
   instructions. `theata_native_ideal.stim` holds the noiseless expansion
   from step 1.

For `main1` at `d=9, p=0.004`, the directory
`results/main1/d9_r9_z/p003_0.004/` contains both Stim files and the
hardware summary `hardware.json`.


## English documentation update

The package's human-readable text was audited and already contained no
Japanese/CJK text, including notebook sources and outputs. The documentation
and notebook now include a general mechanism-index derivation and a precise
support-rank condition. No estimator, sampler, decoder, numerical result, or
notebook code cell was changed in this documentation update.

## Executed validation

- Python 3.12.14, Stim 1.16.0, PyMatching 2.4.0. Exact dependency versions: `requirements-tested.txt`.
- Full suite: 41 passed in 3.67 seconds. One external pytest-plugin deprecation warning; no package test failures.
- Both X- and Z-memory native Clifford rewrites checked; exact population-moment recovery, boundary inversion, identifiability guards, integer-overflow protection, oracle isolation, worker invariance and resume consistency tested.
- Shell entry point executed on the complete `configs/demo.yaml` sweep with four workers.
- The d=7 production hardware model was compiled and both hardware validators passed. All five configured physical rates project to 883 identifiable edges with zero LUT clipping; the maximum selected marginal at p=0.004 is 0.454385. The million-shot production sweep was not executed.
- Walkthrough: all seven code cells executed successfully with IPython InteractiveShell, including multiprocessing and rich image output. This host prohibits kernel sockets, so notebook validation used the provided socket-free runner; a separate Jupyter server launch was not tested.

## Demo protocol

- Distances 3 and 5; Z memory; rounds equal distance.
- Physical p = mean total nonidentity error probability at physical MS locations.
- Native quantum order is conservative and retained from the Stim standard generator. THeatA supplies transport scheduling, heating and noise.
- THeatA cooling threshold 1.5, strict lookup validation. Zero LUT clipping at both distances.
- One independent calibration dataset of 100,000 shots per distance/rate, with prefixes 1,000 and 10,000.
- 50,000 independent heldout shots per distance/rate. All compared decoders share those heldout shots.
- Full actual DEM sampling; the inference and matching graph is its exact identifiable memory-sector marginal.
- LER is measured per complete memory experiment.

## Observed logical error rates

| d | Mean MS p | Actual prior | Estimated N=1,000 | Estimated N=10,000 | Estimated N=100,000 |
|---|---|---|---|---|---|
| 3 | 0.0005 | 0.00118 | 0.00572 | 0.00246 | 0.00116 |
| 3 | 0.001 | 0.0046 | 0.0048 | 0.00444 | 0.00462 |
| 3 | 0.002 | 0.01686 | 0.02042 | 0.01682 | 0.01696 |
| 3 | 0.004 | 0.057 | 0.05798 | 0.057 | 0.0569 |
| 5 | 0.0005 | 0.00696 | 0.00872 | 0.00724 | 0.00692 |
| 5 | 0.001 | 0.04158 | 0.04884 | 0.04256 | 0.04178 |
| 5 | 0.002 | 0.18592 | 0.21368 | 0.18832 | 0.1864 |
| 5 | 0.004 | 0.42704 | 0.44762 | 0.43126 | 0.42716 |

CSV files retain counts, Wilson intervals and paired discordance statistics. The displayed rates are direct observations from this one calibration repetition; they are not pooled across dependent fitted models.

The larger distance has higher LER in this hardware configuration. The compiled duration is 0.457635 s at d=3 and 7.378033 s at d=5, with all original ancillas retained and a conservative single-LIZ native order. These results evaluate prior estimation for that specified hardware schedule; they are not a surface-code threshold study under uniform circuit noise.

The raw p_ij matrix is retained, including negative empirical estimates. Decoder priors use fixed clipping at 1e-8; per-fit diagnostics identify the affected edges.
