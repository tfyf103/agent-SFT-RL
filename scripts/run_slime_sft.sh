#!/usr/bin/env bash
# Linux, inside an isolated CUDA container. GPU integration has NOT been run.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SLIME_DIR:?Set the path to the pinned Slime checkout}"
: "${MEGATRON_DIR:?Set the compatible Megatron-LM path from your Slime image}"
: "${HF_MODEL:?Set the local pinned Qwen3-4B-Instruct-2507 model directory}"
: "${MCORE_INIT:?Set the initial converted Megatron checkpoint directory}"
: "${SFT_DATA:?Set the generated training messages JSONL path}"
: "${OUTPUT_DIR:?Set a new, dedicated SFT output directory}"
: "${RAY_DASHBOARD_ADDRESS:?Set the dashboard URL of YOUR isolated Ray cluster}"
NUM_GPUS="${NUM_GPUS:-4}"
SEED="${SEED:-42}"
EXPECTED_SLIME=8c17b676cb57af1d17ee4402e91e9209af84b60b
[[ "$(git -C "$SLIME_DIR" rev-parse HEAD)" == "$EXPECTED_SLIME" ]] || {
  echo "Wrong Slime revision; see upstream.lock.json." >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Use a new OUTPUT_DIR to avoid accidental resume." >&2; exit 2; }
[[ -f "$SFT_DATA" && -d "$HF_MODEL" && -d "$MCORE_INIT" ]] || exit 2
python - "$SFT_DATA" <<'PY'
import json, sys
n = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("metadata", {}).get("split")
        if split is None:
            split = row.get("metadata", {}).get("task", {}).get("split")
        if split != "train":
            raise SystemExit("SFT only accepts explicit metadata split=train.")
        if not any(m.get("role") == "assistant" for m in row["messages"]):
            raise SystemExit("A trajectory contains no assistant targets.")
        n += 1
assert n > 0, "Empty SFT data"
print(f"Validated {n} training trajectories.")
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
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL" --ref-load "$MCORE_INIT" \
  --save "$OUTPUT_DIR/checkpoints" --save-interval 10 \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --prompt-data "$SFT_DATA" --input-key messages --metadata-key metadata \
  --rollout-shuffle --num-epoch 2 \
  --rollout-batch-size 16 --n-samples-per-prompt 1 --global-batch-size 16 \
  --loss-type sft_loss --loss-mask-type qwen3 --calculate-per-token-loss \
  --disable-compute-advantages-and-returns --debug-train-only \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 \
  --lr-warmup-fraction 0.1 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --bf16 --seed "$SEED" \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash 2>&1 | tee "$OUTPUT_DIR/train.log"
