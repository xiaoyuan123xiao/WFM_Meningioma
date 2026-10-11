#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:?"Set DATA_DIR to the training directory"}
OUT_ROOT=${OUT_ROOT:-runs/wfm_meningioma}
SEEDS=${SEEDS:-42}
IMPL=${IMPL:-paper}
SSIM_WEIGHT=${SSIM_WEIGHT:-0.0}
HF_PRED=${HF_PRED:-pred_target}
HF_TIME=${HF_TIME:-zero}
EXCLUDE_CASES=${EXCLUDE_CASES:-}

DISC_ARGS=""
EXCLUDE_ARGS=""
if [[ -n "$EXCLUDE_CASES" ]]; then
  EXCLUDE_ARGS="--exclude_cases $EXCLUDE_CASES"
fi
if [[ "$IMPL" == "legacy_release" ]]; then
  DISC_ARGS="--discrepancy_target lll_abs --discrepancy_input target"
fi

for variant in original highfreq_pure dagcn anatomy; do
  for steps in 200 500 1000; do
    for seed in $SEEDS; do
      run_dir="$OUT_ROOT/${variant}_${steps}_seed${seed}"
      echo "=== variant=$variant steps=$steps seed=$seed ==="
      python scripts/train_wfm_ablation.py \
        --variant "$variant" \
        --data_dir "$DATA_DIR" \
        --output_dir "$run_dir" \
        --seed "$seed" \
        --max_iterations "$steps" \
        --ssim_weight "$SSIM_WEIGHT" \
        --high_freq_prediction "$HF_PRED" \
        --high_freq_time "$HF_TIME" \
        $DISC_ARGS \
        $EXCLUDE_ARGS
    done
  done
done








if [[ "${RUN_INDEPENDENT:-0}" == "1" ]]; then
  for variant in diff anat hf_anat; do
    for steps in 200 500 1000; do
      for seed in $SEEDS; do
        run_dir="$OUT_ROOT/${variant}_${steps}_seed${seed}"
        echo "=== independent variant=$variant steps=$steps seed=$seed ==="
        python scripts/train_wfm_ablation.py \
          --variant "$variant" \
          --data_dir "$DATA_DIR" \
          --output_dir "$run_dir" \
          --seed "$seed" \
          --max_iterations "$steps" \
          --ssim_weight "$SSIM_WEIGHT" \
          --high_freq_prediction "$HF_PRED" \
          --high_freq_time "$HF_TIME" \
          $DISC_ARGS \
          $EXCLUDE_ARGS
      done
    done
  done
fi
