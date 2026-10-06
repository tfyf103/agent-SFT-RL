"""Single-device multi-turn GRPO; honest reference implementation, not Slime."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bank_agent.training import (
    add_model_arguments, append_jsonl, dependency_versions, grpo_update,
    load_model, load_tokenizer, restore_checkpoint, rollout, save_checkpoint,
    seed_everything, sha256_file, validate_training_tasks,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--updates-per-group", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    if min(args.groups, args.updates_per_group, args.max_turns, args.max_new_tokens,
           args.max_context, args.save_every) < 1:
        parser.error("Counts and token limits must be positive")
    if args.learning_rate <= 0 or args.beta < 0 or not 0 < args.clip_epsilon < 1:
        parser.error("Invalid optimizer/PPO settings")

    import torch
    from bank_agent.env import BankEnv, load_tasks
    tasks = load_tasks(args.tasks)
    validate_training_tasks(tasks)
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args)
    policy = load_model(args, resume=args.resume)
    # Always reload the ORIGINAL initialization, including its initial adapter.
    # A resumed policy must never silently become its own reference.
    reference = load_model(args, reference=True)
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.0
    )
    configuration = vars(args) | {
        "trainer": "multi_turn_grpo", "tasks_sha256": sha256_file(args.tasks),
        "sampling": "temperature=1, top_k=0, top_p=1, no repetition penalty",
        "normalization": "population reward std; per-trajectory assistant-token mean",
    }
    state = {"completed_groups": 0, "optimizer_steps": 0}
    if args.resume:
        state = restore_checkpoint(
            optimizer, args.resume, configuration, {"resume", "output", "groups"}
        )
    (output / "run_metadata.json").write_text(json.dumps({
        "configuration": configuration, "dependencies": dependency_versions(),
        "task_count": len(tasks),
        "result_scope": "Training rewards only; report held-out agent evaluation separately",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for group_index in range(state["completed_groups"], args.groups):
        task = random.choice(tasks)
        trajectories = [
            rollout(
                policy, tokenizer, task, BankEnv, max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens, max_context=args.max_context,
            ) for _ in range(args.group_size)
        ]
        append_jsonl(output / "rollouts.jsonl", {
            "group": group_index + 1, "task_id": task.get("id", task.get("task_id")),
            "trajectories": [asdict(trajectory) for trajectory in trajectories],
        })
        for update in range(args.updates_per_group):
            metrics = grpo_update(
                policy, reference, optimizer, trajectories, beta=args.beta,
                clip_epsilon=args.clip_epsilon, max_grad_norm=args.max_grad_norm,
            )
            state["optimizer_steps"] += int(metrics["assistant_tokens"] > 0)
            metrics.update({
                "group": group_index + 1, "update": update + 1,
                "optimizer_steps": state["optimizer_steps"],
            })
            append_jsonl(output / "metrics.jsonl", metrics)
            print(json.dumps(metrics), flush=True)
            if not metrics["reward_learning_signal"]:
                print("NO_REWARD_LEARNING_SIGNAL: all group advantages are zero; "
                      "only a nonzero reference KL could contribute a gradient.", flush=True)
        state["completed_groups"] = group_index + 1
        if (group_index + 1) % args.save_every == 0 or group_index + 1 == args.groups:
            save_checkpoint(
                policy, tokenizer, optimizer, output / f"group-{group_index + 1:06d}",
                state, configuration,
            )
    print(f"Checkpoints: {output.resolve()}. Training reward is not a held-out benchmark result.")


if __name__ == "__main__":
    main()
