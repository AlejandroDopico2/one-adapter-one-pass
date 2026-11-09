#!/usr/bin/env bash
# Post-processing of finished runs: classifier staleness (Fig. 1, Fig. 2b), inference cost (Table 2) and
# the multi-pass inference budget of TUNA. Writes to results/ and expects scripts/run_experiments.sh to be done.
#
#   DATA_ROOT=/path/to/data DEVICE=0 bash scripts/postprocess.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
DEVICE=${DEVICE:-0}
PY=${PY:-python}
mkdir -p results

# Fig. 1: staleness S = A_refit - A_T of the shared adapter (refit oracle on the final features).
: > results/std_refit.log
for r in runs/std/inr200_in21k/{none,nonecomp,distill,distcomp}_s* runs/std/cifar_in21k/{nonecomp,distill,distcomp}_s*; do
  $PY scripts/head_refit.py --run-dir "$r" --data-root "$DATA_ROOT" --device "$DEVICE" --out results/std_refit.json >> results/std_refit.log
done

# Fig. 2b: staleness of the domain-isolated oracle on VTAB.
$PY scripts/srw_refit.py runs/std/vtab_in21k/* > results/vtab_oracle_staleness.log

# Table 2: parameters, storage, GFLOPs, latency and energy of one inference (run alone on the GPU).
$PY scripts/cost_benchmark.py runs/std/inr200_in21k/{distcomp,ease,tuna,sema}_s1993 \
  --device "$DEVICE" --out results/cost_inr200_in21k.json
$PY scripts/cost_benchmark.py runs/fair/inr200_in21k/{tuna,ease,sema}_b3_s1993 \
  --device "$DEVICE" --out results/cost_fair_inr200_in21k.json

# Accuracy of TUNA/EASE restricted to fewer forward passes.
$PY scripts/inference_budget.py --device "$DEVICE" --out results/inference_budget_fair.json \
  --run-dir runs/fair/*/tuna_b3_s*

# Tables and figures of the paper.
$PY paper/make_figures.py
