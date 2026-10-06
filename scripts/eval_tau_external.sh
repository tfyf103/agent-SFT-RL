#!/usr/bin/env bash
# External benchmark only. No training data are taken from banking_knowledge.
set -euo pipefail
: "${TAU_DIR:?Set the pinned tau2-bench checkout directory}"
: "${AGENT_LLM:?Set a LiteLLM model name or local OpenAI-compatible agent model}"
: "${USER_LLM:?Set an independent, fixed user simulator model}"
EXPECTED_TAU=5bfa7e37b36656b37dc6d022156be6563c1007f3
[[ "$(git -C "$TAU_DIR" rev-parse HEAD)" == "$EXPECTED_TAU" ]] || {
  echo "Wrong tau2-bench revision." >&2; exit 2; }
# Optional JSON args support local api_base; keep API keys in environment variables.
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-"{}"}"
USER_LLM_ARGS="${USER_LLM_ARGS:-"{}"}"
TRIALS="${TRIALS:-4}"
RUN_NAME="${RUN_NAME:-bank_external_$(date -u +%Y%m%dT%H%M%SZ)}"
EXTRA=()
# NUM_TASKS=2 is an initial connectivity check, never a full benchmark claim.
if [[ -n "${NUM_TASKS:-}" ]]; then EXTRA+=(--num-tasks "$NUM_TASKS"); fi
cd "$TAU_DIR"
uv run --frozen --extra knowledge tau2 run \
  --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" --user-llm-args "$USER_LLM_ARGS" \
  --num-trials "$TRIALS" --max-concurrency "${MAX_CONCURRENCY:-2}" \
  --seed 300 --max-steps 100 --max-retries 0 \
  --save-to "$RUN_NAME" --verbose-logs "${EXTRA[@]}"
