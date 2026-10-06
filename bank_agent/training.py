"""Single-device reference training for JSON-action bank agents.

Only standard-library modules are imported at module import time, so token-mask,
advantage and resume-format helpers can be tested without a GPU or PyTorch.
Training is intentionally transparent, not a distributed/high-throughput trainer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100


class ExampleTooLong(ValueError):
    pass


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checked_messages(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    cleaned = []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Use user-role messages with an explicit environment label for observations")
        if not isinstance(message.get("content"), str):
            raise ValueError("Only string content is supported")
        cleaned.append({"role": message["role"], "content": message["content"]})
    return cleaned


def chat_ids(tokenizer: Any, messages: list[dict], generation: bool = False) -> list[int]:
    ids = tokenizer.apply_chat_template(
        checked_messages(messages), tokenize=True, add_generation_prompt=generation
    )
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise TypeError("Tokenizer chat template must return a flat token-id list")
    return ids


def sft_example(tokenizer: Any, messages: list[dict], max_length: int) -> dict:
    """Mask all context and supervise only assistant completion spans.

    A template is accepted only when each generation prompt and completed turn
    is an exact token prefix of the final serialized conversation. Templates
    that rewrite earlier turns are rejected rather than silently mislabelled.
    Overlong examples are rejected whole; actions are never cut in half.
    """
    messages = checked_messages(messages)
    ids = chat_ids(tokenizer, messages)
    if len(ids) > max_length:
        raise ExampleTooLong(f"{len(ids)} tokens exceeds max_length={max_length}")
    labels = [IGNORE_INDEX] * len(ids)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index == 0:
            raise ValueError("An assistant turn must have preceding context")
        prompt = chat_ids(tokenizer, messages[:index], generation=True)
        completed = chat_ids(tokenizer, messages[:index + 1])
        if completed[:len(prompt)] != prompt or ids[:len(completed)] != completed:
            raise ValueError("Chat template is not prefix-stable; unsupported for assistant-only masking")
        if len(completed) <= len(prompt):
            raise ValueError("Empty assistant completion after chat-template tokenization")
        labels[len(prompt):len(completed)] = ids[len(prompt):len(completed)]
        spans.append([len(prompt), len(completed)])
    if not spans or not any(x != IGNORE_INDEX for x in labels[1:]):
        raise ValueError("No trainable assistant token")
    return {"input_ids": ids, "labels": labels, "assistant_spans": spans}


def record_partition(record: dict) -> tuple[str | None, str | None]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("task", {}), dict):
        raise ValueError("metadata and metadata.task must be objects")
    sources = [record, metadata, metadata.get("task", {})]
    splits = {item["split"] for item in sources if item.get("split") is not None}
    identifiers = {item["task_id"] for item in sources if item.get("task_id") is not None}
    if len(splits) > 1 or len(identifiers) > 1:
        raise ValueError("Conflicting split or task_id metadata")
    if any(not isinstance(value, str) or not value for value in splits | identifiers):
        raise ValueError("Split/task_id metadata must be nonempty strings")
    return next(iter(splits), None), next(iter(identifiers), None)


def assert_sft_disjoint(training: list[dict], validation: list[dict]) -> None:
    for key in ("task_id", "messages_sha256"):
        left = {row[key] for row in training if row.get(key) is not None}
        right = {row[key] for row in validation if row.get(key) is not None}
        if left & right:
            raise ValueError(f"Training/validation overlap on {key}")


def validate_training_tasks(tasks: list[dict]) -> None:
    if not tasks:
        raise ValueError("Training tasks must be nonempty")
    identifiers = set()
    for task in tasks:
        if task.get("split") != "train":
            raise ValueError("GRPO accepts only explicit split=train tasks")
        identifier = task.get("task_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every training task needs a nonempty task_id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate training task_id: {identifier}")
        identifiers.add(identifier)


def load_sft_examples(path: str, tokenizer: Any, max_length: int, *,
                      allowed_splits: tuple[str, ...] = ("train",)) -> tuple[list[dict], dict]:
    examples, too_long, total, unlabeled = [], 0, 0, 0
    split_counts = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
                split, identifier = record_partition(record)
                if split is not None and split not in allowed_splits:
                    raise ValueError(f"Disallowed split={split}; expected {allowed_splits}")
                if split is None:
                    unlabeled += 1
                else:
                    split_counts[split] = split_counts.get(split, 0) + 1
                example = sft_example(tokenizer, record["messages"], max_length)
                example["task_id"] = identifier
                example["messages_sha256"] = hashlib.sha256(json.dumps(
                    checked_messages(record["messages"]), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                examples.append(example)
            except ExampleTooLong:
                too_long += 1
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError("No usable SFT examples")
    return examples, {"total": total, "used": len(examples), "overlength_rejected": too_long,
                      "unlabeled_split": unlabeled, "split_counts": split_counts}


def padded_batch(examples: list[dict], pad_token_id: int, device: str) -> dict:
    import torch
    length = max(len(example["input_ids"]) for example in examples)
    return {
        "input_ids": torch.tensor([
            x["input_ids"] + [pad_token_id] * (length - len(x["input_ids"]))
            for x in examples
        ], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([
            [1] * len(x["input_ids"]) + [0] * (length - len(x["input_ids"]))
            for x in examples
        ], dtype=torch.long, device=device),
        "labels": torch.tensor([
            x["labels"] + [IGNORE_INDEX] * (length - len(x["labels"]))
            for x in examples
        ], dtype=torch.long, device=device),
    }


def parse_action(text: str) -> dict:
    """Strict JSON: malformed outputs become observable environment failures."""
    try:
        action = json.loads(text.strip())
    except (ValueError, TypeError):
        return {"tool": "__invalid_action__", "arguments": {"error": "invalid_json"}}
    if (not isinstance(action, dict) or set(action) != {"tool", "arguments"}
            or not isinstance(action["tool"], str) or not isinstance(action["arguments"], dict)):
        return {"tool": "__invalid_action__", "arguments": {"error": "invalid_schema"}}
    return action


def group_advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    """Population-standardized advantages; identical rewards give zero."""
    if len(rewards) < 2:
        raise ValueError("GRPO requires group_size >= 2")
    if epsilon <= 0 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Rewards must be finite and epsilon positive")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance) + epsilon
    return [(reward - mean) / scale for reward in rewards]


def clipped_objective(ratio: float, advantage: float, epsilon: float) -> float:
    """Scalar counterpart of the differentiable PPO clipping expression."""
    return min(ratio * advantage, max(1 - epsilon, min(1 + epsilon, ratio)) * advantage)


def k3_kl(logp: float, reference_logp: float) -> float:
    delta = reference_logp - logp
    return math.expm1(delta) - delta


@dataclass
class Turn:
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]


@dataclass
class Trajectory:
    turns: list[Turn]
    reward: float
    success: bool
    stop_reason: str
    result: dict

    @property
    def tokens(self) -> int:
        return sum(len(turn.completion_ids) for turn in self.turns)


def completion_logprobs(model: Any, prompt_ids: list[int], completion_ids: list[int]):
    """Teacher-force the exact sampled token sequence; no re-tokenization."""
    import torch
    if not prompt_ids or not completion_ids:
        raise ValueError("Both prompt and completion tokens are required")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = output.logits[0, len(prompt_ids) - 1:-1, :].float()
    targets = ids[0, len(prompt_ids):]
    return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def eos_ids(model: Any, tokenizer: Any) -> list[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("An explicit EOS token is required")
    return eos if isinstance(eos, list) else [eos]


def rollout(model: Any, tokenizer: Any, task: dict, env_factory: Any, *,
            max_turns: int, max_new_tokens: int, max_context: int) -> Trajectory:
    """Sample one full environment episode under an unchanged evaluation-mode policy.

    Every sampled token (including EOS) gets an old-policy log probability.
    Environment observations only enter subsequent prompts. Truncated episodes
    remain in the group with zero reward, including episodes with no tokens.
    """
    import torch
    from transformers import GenerationConfig
    env = env_factory(copy.deepcopy(task))
    observation = env.reset()
    turns, accumulated_reward = [], 0.0
    stop_reason = "max_turns"
    model.eval()  # dropout must be off for rollout and likelihood recomputation
    endings = eos_ids(model, tokenizer)
    for _ in range(max_turns):
        messages = getattr(env, "messages", None)
        if messages is None:
            messages = observation["messages"]
        prompt = chat_ids(tokenizer, messages, generation=True)
        remaining = max_context - len(prompt)
        if remaining <= 0:
            stop_reason = "context_limit"
            break
        limit = min(max_new_tokens, remaining)
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        # Build from scratch: inherited model defaults such as top_k or
        # repetition penalties would invalidate the stored policy logprobs.
        config = GenerationConfig(
            do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
            typical_p=1.0, repetition_penalty=1.0, num_beams=1,
            max_new_tokens=limit, min_new_tokens=0,
            eos_token_id=endings, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                generation_config=config,
            )[0, len(prompt):].tolist()
            old = completion_logprobs(model, prompt, generated).cpu().tolist() if generated else []
        if generated:
            turns.append(Turn(prompt, generated, old))
        if not generated or generated[-1] not in endings:
            stop_reason = "generation_limit"
            break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        action = parse_action(text)
        observation, reward, terminated, _ = env.step(action)
        if not math.isfinite(float(reward)):
            raise ValueError("Environment produced a non-finite reward")
        accumulated_reward += float(reward)
        if terminated:
            stop_reason = "terminated"
            break
    result = env.result()
    if stop_reason == "terminated" and result.get("truncated"):
        stop_reason = "environment_limit"
    truncated = stop_reason != "terminated"
    return Trajectory(
        turns, 0.0 if truncated else accumulated_reward,
        bool(result.get("success", False)) and not truncated, stop_reason, result,
    )


def grpo_update(policy: Any, reference: Any, optimizer: Any, trajectories: list[Trajectory],
                *, beta: float, clip_epsilon: float, max_grad_norm: float = 1.0) -> dict:
    """One PPO-style update; each group member has fixed weight 1/group_size.

    Each trajectory's token losses are averaged over its assistant tokens.
    A zero-token/truncated member is never removed from the group denominator.
    Per-turn backward frees activations early. No critic or value model is used.
    """
    import torch
    advantages = group_advantages([trajectory.reward for trajectory in trajectories])
    policy.eval()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_kl, weighted_tokens = 0.0, 0.0, 0
    group_size = len(trajectories)
    for trajectory, advantage in zip(trajectories, advantages):
        if trajectory.tokens == 0:
            continue
        for turn in trajectory.turns:
            current = completion_logprobs(policy, turn.prompt_ids, turn.completion_ids)
            with torch.no_grad():
                ref = completion_logprobs(reference, turn.prompt_ids, turn.completion_ids)
            old = torch.tensor(turn.old_logprobs, dtype=current.dtype, device=current.device)
            if old.shape != current.shape:
                raise ValueError("Stored rollout logprobs do not match completion tokens")
            ratio = (current - old).exp()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
            delta = ref - current
            kl = delta.exp() - delta - 1
            token_loss = -torch.minimum(unclipped, clipped) + beta * kl
            loss = token_loss.sum() / trajectory.tokens / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO loss; checkpoint previous stable state")
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach().sum())
            weighted_tokens += len(turn.completion_ids)
    if weighted_tokens:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    return {
        "loss": total_loss,
        "mean_token_kl": total_kl / max(weighted_tokens, 1),
        "mean_reward": sum(x.reward for x in trajectories) / group_size,
        "success_rate": sum(x.success for x in trajectories) / group_size,
        "zero_advantage_fraction": sum(abs(x) < 1e-12 for x in advantages) / group_size,
        "reward_learning_signal": any(abs(x) >= 1e-12 for x in advantages),
        "truncated": sum(x.stop_reason != "terminated" for x in trajectories),
        "empty_trajectories": sum(x.tokens == 0 for x in trajectories),
        "assistant_tokens": weighted_tokens,
    }


def add_model_arguments(parser: Any) -> None:
    parser.add_argument("--model", required=True, help="Base/SFT model directory or Hugging Face ID")
    parser.add_argument("--revision", default=None, help="Pin remote model to a commit for reproducibility")
    parser.add_argument("--adapter", default=None, help="Optional initial PEFT adapter")
    parser.add_argument("--lora", action="store_true", help="Create a fresh LoRA if no adapter is supplied")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default=None, help="Trusted checkpoint produced by this trainer")


def load_tokenizer(args: Any):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.offline,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError("Model must supply a chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad or EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args: Any, *, reference: bool = False, resume: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    # This reference trainer deliberately avoids unscaled fp16 training.
    if dtype == "float16" and not reference:
        raise ValueError("Use bfloat16 or float32; fp16 GradScaler is not implemented")
    uses_adapter = bool(args.adapter or args.lora)
    location = args.model if reference or not resume or uses_adapter else str(Path(resume) / "model")
    model = AutoModelForCausalLM.from_pretrained(
        location, revision=args.revision if location == args.model else None,
        local_files_only=args.offline, trust_remote_code=False,
        torch_dtype=getattr(torch, dtype), attn_implementation="eager",
    ).to(device)
    adapter = args.adapter
    if resume and not reference and uses_adapter:
        adapter = str(Path(resume) / "model")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=not reference)
    elif args.lora and not reference:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0, target_modules="all-linear",
        ))
    if reference:
        model.requires_grad_(False)
        model.eval()
    return model


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(model: Any, tokenizer: Any, optimizer: Any, output: Path,
                    state: dict, configuration: dict) -> None:
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    torch.save({
        "optimizer": optimizer.state_dict(), "state": state,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, output / "training_state.pt")
    (output / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_resume(configuration: dict, saved: dict, mutable: set[str]) -> None:
    differences = [
        key for key in set(configuration) | set(saved)
        if key not in mutable and configuration.get(key) != saved.get(key)
    ]
    if differences:
        raise ValueError("Resume configuration differs: " + ", ".join(sorted(differences)))


def restore_checkpoint(optimizer: Any, directory: str, configuration: dict,
                       mutable: set[str]) -> dict:
    import torch
    folder = Path(directory)
    saved_config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    validate_resume(configuration, saved_config, mutable)
    # Optimizer/RNG state uses pickle: only load checkpoints you created/trust.
    saved = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available() and saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["state"]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for package in ("torch", "transformers", "peft"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
