"""Slime rollout for the synthetic BankEnv, pinned to upstream.lock.json.

Status: contract-tested with a fake sampler; GPU/A800 integration NOT executed.
Model-specific wire protocol: Qwen3-4B-Instruct-2507, plain JSON actions.
Interface references (Apache-2.0): THUDM/slime @
8c17b676cb57af1d17ee4402e91e9209af84b60b, examples/search-r1 and utils/types.py.
This adapter is independently written; model tokens are never re-tokenized.
"""
from __future__ import annotations

import json
from typing import Any

from bank_agent.env import BankEnv

MAX_NEW_TOKENS_PER_TURN = 256


def validate_task(metadata: dict, evaluation: bool) -> dict:
    task = metadata.get("task")
    if not isinstance(task, dict):
        raise ValueError("Each rollout row needs metadata.task containing the full task.")
    split = task.get("split")
    permitted = {"valid", "dev", "validation", "test", "stress"} if evaluation else {"train"}
    if split not in permitted:
        raise ValueError(f"Task split {split!r} is not allowed for evaluation={evaluation}.")
    if not isinstance(task.get("task_id"), str):
        raise ValueError("Task must have a stable string task_id.")
    return task


def parse_action(text: str) -> dict:
    # Parsing may strip EOS; the original sampled IDs/logprobs remain unchanged.
    text = text.rstrip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    try:
        action = json.loads(text)
    except (ValueError, TypeError):
        return {"tool": "__invalid__", "arguments": {}}
    return action if isinstance(action, dict) else {"tool": "__invalid__", "arguments": {}}


def observation_suffix(tokenizer: Any, generated_ids: list[int], content: str) -> list[int]:
    """Encode ONLY newly injected environment/header text, never model output."""
    end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(end_ids) != 1 or len(start_ids) != 1:
        raise ValueError("Adapter requires Qwen ChatML special tokens.")
    # Stop at im_end should preserve that sampled token. A sampler which omits it
    # receives a non-trainable structural delimiter; no synthetic model logprob.
    close = "" if generated_ids and generated_ids[-1] == end_ids[0] else "<|im_end|>"
    suffix = close + "\n<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(suffix, add_special_tokens=False)


async def generate(args, sample, sampling_params: dict, evaluation: bool = False):
    """Generate one trajectory and mutate the original Sample to retain identity."""
    # Lazy imports keep protocol helpers inspectable without installing Slime.
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    if getattr(args, "partial_rollout", False):
        raise ValueError("Bank adapter does not support partial rollout.")
    if getattr(args, "use_score_centering", False):
        raise ValueError("Score-centering is not supported by this adapter.")
    if sample.response_length:
        raise ValueError("Fresh trajectories only; existing response would lose alignment.")
    task = validate_task(sample.metadata, evaluation)
    env = BankEnv(task)
    tokenizer = GenerateState(args).tokenizer
    prompt = tokenizer.apply_chat_template(env.messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("<|im_start|>assistant\n"):
        raise ValueError("Only the pinned non-thinking Qwen3 Instruct template is supported.")
    sample.prompt = prompt
    sample.tokens = tokenizer.encode(prompt, add_special_tokens=False)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.reward = None
    sample.status = Sample.Status.PENDING
    prompt_length = len(sample.tokens)

    # The cap counts both generated tokens and inserted observations.
    budget = int(sampling_params.get("max_new_tokens") or 8192)
    if budget < 1:
        raise ValueError("Response budget must be positive.")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    for _ in range(env.max_steps):
        remaining = budget - sample.response_length
        if remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        params = dict(sampling_params)
        params.update(max_new_tokens=min(MAX_NEW_TOKENS_PER_TURN, remaining),
                      no_stop_trim=True, skip_special_tokens=False)
        # Use token IDs as the next prompt: no decode/re-tokenize round trip.
        output = await post(url, {"input_ids": list(sample.tokens),
                                  "sampling_params": params, "return_logprob": True})
        meta = output["meta_info"]
        finish = meta["finish_reason"]["type"]
        if finish == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        rows = meta.get("output_token_logprobs")
        if not rows:
            raise RuntimeError("Sampler must return nonempty output_token_logprobs.")
        ids = [int(row[1]) for row in rows]
        logps = [float(row[0]) for row in rows]
        raw_text = output["text"]
        sample.append_response_tokens(
            args, tokens=ids, log_probs=logps, trainable=True,
            meta_info=meta, text=raw_text, update_terminal_info=False)
        if finish == "length":
            # Never execute a truncated JSON action.
            sample.status = Sample.Status.TRUNCATED
            break
        if finish != "stop":
            raise RuntimeError(f"Unsupported sampler finish reason: {finish}")
        _, _, done, result = env.step(parse_action(raw_text))
        if done:
            sample.status = Sample.Status.TRUNCATED if result["truncated"] else Sample.Status.COMPLETED
            break
        new_ids = observation_suffix(tokenizer, ids, env.messages[-1]["content"])
        if sample.response_length + len(new_ids) >= budget:
            sample.status = Sample.Status.TRUNCATED
            break
        sample.append_response_tokens(
            args, tokens=new_ids, trainable=False,
            text=tokenizer.decode(new_ids, skip_special_tokens=False),
            update_terminal_info=False)
    else:
        sample.status = Sample.Status.TRUNCATED

    result = env.result()
    sample.metadata["bank_result"] = result
    sample.metadata["bank_trace"] = env.trace
    sample.metadata["bank_evaluation"] = evaluation
    sample.reward = float(result["success"])
    if sample.status == Sample.Status.TRUNCATED:
        sample.reward = 0.0
    assert sample.response_length == len(sample.tokens) - prompt_length
    assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
    assert all(p == 0.0 for p, m in zip(sample.rollout_log_probs, sample.loss_mask) if not m)
    return sample


async def reward(args, sample, **kwargs) -> float:
    """Outcome reward, shared across all policies. No extra credit for refusal."""
    result = sample.metadata.get("bank_result")
    if result is None:
        raise ValueError("Missing verified bank_result; reward cannot be inferred from prose.")
    return float(result["success"])
