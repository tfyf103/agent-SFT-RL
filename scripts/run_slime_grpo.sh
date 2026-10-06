#!/usr/bin/env bash
# Intended 4-GPU colocated Slime run. GPU/A800 compatibility is unverified.
# Does NOT terminate shared processes or create/alter a shared Ray cluster.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SLIME_DIR:?Set the pinned Slime checkout path}"
: "${MEGATRON_DIR:?Set the compatible Megatron-LM path}"
: "${HF_MODEL:?Set the exported SFT HF checkpoint path}"
: "${MCORE_INIT:?Set the Megatron conversion of that SAME SFT checkpoint}"
: "${TRAIN_DATA:?Set rollout training JSONL path}"
: "${DEV_DATA:?Set separate rollout dev JSONL path}"
: "${OUTPUT_DIR:?Set a new, dedicated GRPO output directory}"
: "${RAY_DASHBOARD_ADDRESS:?Set the dashboard URL of YOUR isolated Ray cluster}"
NUM_GPUS="${NUM_GPUS:-4}"
SEED="${SEED:-42}"
EXPECTED_SLIME=8c17b676cb57af1d17ee4402e91e9209af84b60b
[[ "$(git -C "$SLIME_DIR" rev-parse HEAD)" == "$EXPECTED_SLIME" ]] || exit 2
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Use a new OUTPUT_DIR." >&2; exit 2; }
[[ -d "$HF_MODEL" && -d "$MCORE_INIT" ]] || exit 2
python - "$TRAIN_DATA" "$DEV_DATA" <<'PY'
import json, sys
groups = []
for filename, split in zip(sys.argv[1:], ("train", "valid")):
    with open(filename, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows, "Empty rollout split"
    tasks = [row["metadata"]["task"] for row in rows]
    assert all(t["split"] == split for t in tasks), "Mixed or wrong split"
    ids = {t["task_id"] for t in tasks}
    assert len(ids) == len(tasks), "Duplicate task IDs"
    groups.append(ids)
assert groups[0].isdisjoint(groups[1]), "Training/dev task leakage"
print("Training/dev identifiers are disjoint.")
PY
mkdir -p "$OUTPUT_DIR"
git -C "$SLIME_DIR" rev-parse HEAD > "$OUTPUT_DIR/slime_commit.txt"
python -m pip freeze > "$OUTPUT_DIR/packages.txt"
nvidia-smi > "$OUTPUT_DIR/hardware.txt"
export PYTHONPATH="$SLIME_DIR:$PROJECT_ROOT:$MEGATRON_DIR:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
RUNTIME_ENV_JSON="$(python -c 'import json,os; print(json.dumps({"env_vars":{"PYTHONPATH":os.environ["PYTHONPATH"],"CUDA_DEVICE_MAX_CONNECTIONS":"1"}}))')"
source "$SLIME_DIR/scripts/models/qwen3-4B-Instruct-2507.sh"
ray job submit --address "$RAY_DASHBOARD_ADDRESS" --runtime-env-json "$RUNTIME_ENV_JSON" -- \
  python "$SLIME_DIR/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" \
  --rollout-num-gpus "$NUM_GPUS" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL" --ref-load "$MCORE_INIT" \
  --save "$OUTPUT_DIR/checkpoints" --save-interval 5 \
  --prompt-data "$TRAIN_DATA" --input-key prompt --metadata-key metadata \
  --rollout-shuffle --num-rollout "${NUM_ROLLOUT:-20}" \
  --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16 \
  --rollout-max-response-len 8192 --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --custom-generate-function-path integrations.slime_bank.generate \
  --custom-rm-path integrations.slime_bank.reward \
  --eval-interval 5 --eval-prompt-data bank-dev "$DEV_DATA" \
  --eval-input-key prompt --n-samples-per-eval-prompt 1 \
  --eval-max-response-len 8192 --eval-temperature 0 --eval-top-k 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 \
  --kl-loss-type low_var_kl --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --tensor-model-parallel-size 2 --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --bf16 --seed "$SEED" \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash 2>&1 | tee "$OUTPUT_DIR/train.log"
