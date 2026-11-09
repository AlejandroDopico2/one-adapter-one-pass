#!/usr/bin/env bash
# Runs every experiment behind the paper, sequentially, into runs/<group>/<benchmark>/<name>_s<seed>.
# The run names are what paper/make_figures.py expects, so do not rename them.
#
#   DATA_ROOT=/path/to/data DEVICE=0 bash scripts/run_experiments.sh [main|vtab|all]
#
# `main` : ImageNet-R (10 tasks) and CIFAR-100 (20 tasks): Table 1, Table 2, Fig. 1.
# `vtab` : VTAB (5 tasks), k domains per task, k = 1..5: Table 1 (k=1) and Fig. 2.
# Each run takes hours on one GPU; split the lines across GPUs/lanes if you have several.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with cifar-100-python/, imagenet-r/ and VTAB}
DEVICE=${DEVICE:-0}
SEEDS=${SEEDS:-"1993 1994 1995"}
PY=${PY:-python}
GROUP=${1:-all}

M=configs/methods
B=configs/baselines
BENCH_INR=configs/benchmarks/imagenet_r_b0_inc20_200_in21k.yaml
BENCH_CIFAR=configs/benchmarks/cifar100_b0_inc5_in21k.yaml

run() { # name benchmark method seed
  local name=$1 bench=$2 method=$3 seed=$4
  if ls "runs/$name"/*/metrics.json >/dev/null 2>&1; then echo "skip $name"; return; fi
  $PY train.py --benchmark "$bench" --method "$method" --data-root "$DATA_ROOT" \
    --output-root "runs/$name" --seed "$seed" --device "$DEVICE"
}

main() {
  for s in $SEEDS; do
    for bench in "inr200_in21k $BENCH_INR" "cifar_in21k $BENCH_CIFAR"; do
      set -- $bench; d=$1; b=$2
      # ours and its ablation (Fig. 1) + references
      run "std/$d/distcomp_s$s"  "$b" $M/ours.yaml "$s"
      run "std/$d/nonecomp_s$s"  "$b" $M/ablation_comp_only.yaml "$s"
      run "std/$d/distill_s$s"   "$b" $M/ablation_distill_only.yaml "$s"
      run "std/$d/frozen_s$s"    "$b" $M/frozen_after_task1.yaml "$s"
      run "std/$d/noadapt_s$s"   "$b" $M/no_adapter.yaml "$s"
      # modular baselines: matched adapter budget (Table 1) and original recipe (cost study)
      for m in ease sema tuna; do
        run "fair/$d/${m}_b3_s$s" "$b" $B/fair/${m}_b3.yaml "$s"
        run "std/$d/${m}_s$s"     "$b" $B/${m}.yaml "$s"
      done
    done
    run "std/inr200_in21k/none_s$s" "$BENCH_INR" $M/ablation_none.yaml "$s"
  done
}

vtab() {
  for s in $SEEDS; do
    for k in 1 2 3 4 5; do
      b=configs/benchmarks/vtab_conc${k}_in21k.yaml
      run "std/vtab_in21k/k${k}_distcomp_s$s"    "$b" $M/ours.yaml "$s"
      run "std/vtab_in21k/k${k}_distill_s$s"     "$b" $M/ablation_distill_only.yaml "$s"
      run "std/vtab_in21k/k${k}_tuna_s$s"        "$b" $B/fair/tuna_b3.yaml "$s"
      run "std/vtab_in21k/k${k}_ease_s$s"        "$b" $B/ease.yaml "$s"
      run "std/vtab_in21k/k${k}_oracle5_s$s"     "$b" $M/domain_isolated_oracle.yaml "$s"
      run "std/vtab_in21k/k${k}_oracle5comp_s$s" "$b" $M/domain_isolated_oracle_comp.yaml "$s"
    done
    b=configs/benchmarks/vtab_conc1_in21k.yaml  # extra rows of Table 1 (k = 1)
    run "std/vtab_in21k/k1_frozen_s$s"  "$b" $M/frozen_after_task1.yaml "$s"
    run "std/vtab_in21k/k1_noadapt_s$s" "$b" $M/no_adapter.yaml "$s"
    run "std/vtab_in21k/k1_sema_s$s"    "$b" $B/fair/sema_b3.yaml "$s"
    run "std/vtab_in21k/k1_easeb3_s$s"  "$b" $B/fair/ease_b3.yaml "$s"
  done
}

case "$GROUP" in
  main) main ;;
  vtab) vtab ;;
  all) main; vtab ;;
  *) echo "usage: $0 [main|vtab|all]" >&2; exit 2 ;;
esac
