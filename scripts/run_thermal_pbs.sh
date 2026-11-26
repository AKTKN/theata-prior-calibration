#!/usr/bin/env bash
#PBS -N theata-thermal
#PBS -q workq
#PBS -l nodes=1:ppn=96
#PBS -j oe
set -euo pipefail
cd -- "${PBS_O_WORKDIR:?Submit from the project directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
: "${THERMAL_CONFIG:?Pass THERMAL_CONFIG with qsub -v}"
: "${THERMAL_OUT:?Pass THERMAL_OUT with qsub -v}"
exec "${PYTHON_RUN:-.venv/bin/python}" -m theata_prior thermal run \
  --config "${THERMAL_CONFIG}" --out "${THERMAL_OUT}" --workers "${PBS_NP:-96}" --resume
