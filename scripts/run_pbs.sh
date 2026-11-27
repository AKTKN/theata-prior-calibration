#!/usr/bin/env bash
#PBS -N theata-prior
#PBS -q workq
#PBS -l nodes=1:ppn=96
#PBS -j oe
set -euo pipefail
# Submit from the project directory: qsub scripts/run_pbs.sh
cd -- "${PBS_O_WORKDIR:?Submit from the project directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash scripts/run.sh --config configs/production.yaml --out results/production_cooling --workers "${PBS_NP:-96}" --skip-saturated-marginals

