"""Run the CPU toy-policy experiment. NOT an LLM SFT/GRPO reproduction."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from bank_agent.data import generate_tasks
from bank_agent.toy_policy import (
    ACTION_NAMES, FEATURE_NAMES, TinyPolicy, expert_dataset, fit_supervised,
    fit_group_reinforce, policy_rollout,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metrics(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    authorized = [r for r in rows if r["authorized"]]
    unauthorized = [r for r in rows if not r["authorized"]]
    return {
        "n": n, "successes": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_errors": float(np.mean([r["errors"] for r in rows])),
        "false_completion_rate": float(np.mean([r["false_completion"] for r in rows])),
        "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
        "authorized_n": len(authorized),
        "authorized_success_rate": float(np.mean([r["success"] for r in authorized])) if authorized else None,
        "legal_task_false_refusal_rate": float(np.mean([r["false_refusal"] for r in authorized])) if authorized else None,
        "unauthorized_n": len(unauthorized),
        "unauthorized_safe_handoff_rate": float(np.mean([r["success"] for r in unauthorized])) if unauthorized else None,
        "unauthorized_actual_write_count": sum(r["writes"] for r in unauthorized),
    }


def evaluate(model, tasks, output, seed, arm, split):
    rows = []
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            env, _, _ = policy_rollout(model, task)
            row = dict(env.result(), family=task["family"], seed=seed, arm=arm)
            rows.append(row)
            handle.write(json.dumps(dict(row, trace=env.trace), ensure_ascii=False, separators=(",", ":")) + "\n")
    groups = {"overall": rows}
    for role in ("customer", "employee"):
        groups["role:" + role] = [r for r in rows if r["role"] == role]
    for flag in (False, True):
        groups["authorized:" + str(flag).lower()] = [r for r in rows if r["authorized"] == flag]
    for faults in sorted({"+".join(r["faults"]) or "none" for r in rows}):
        groups["faults:" + faults] = [r for r in rows if ("+".join(r["faults"]) or "none") == faults]
    for family in sorted({r["family"] for r in rows}):
        groups["family:" + family] = [r for r in rows if r["family"] == family]
    return [dict(seed=seed, arm=arm, split=split, group=key, **metrics(value)) for key, value in groups.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "toy_policy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--train-tasks", type=int, default=500)
    parser.add_argument("--valid-tasks", type=int, default=200)
    parser.add_argument("--test-tasks", type=int, default=200)
    parser.add_argument("--stress-tasks", type=int, default=200)
    parser.add_argument("--sft-steps", type=int, default=400)
    parser.add_argument("--rl-updates", type=int, default=100)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = {
        "experiment": "CPU toy MLP tool-selection SFT and group-relative REINFORCE",
        "not_llm": True, "not_grpo": True, "device": "CPU", "dtype": "float64",
        "seeds": args.seeds, "hidden": 32, "max_episode_steps": 16,
        "data_counts": {s: getattr(args, s + "_tasks") for s in ("train", "valid", "test", "stress")},
        "sft": {"steps": args.sft_steps, "batch_size": 128, "learning_rate": 0.01},
        "rl": {"updates": args.rl_updates, "groups_per_update": 4, "group_size": 8,
               "learning_rate": 0.001, "reference_kl_beta": 0.02, "reward": "terminal success 0/1"},
        "evaluation": "greedy argmax; final checkpoints; no validation/test checkpoint selection",
        "arguments": "deterministic public-observation binding; not learned",
        "features": list(FEATURE_NAMES), "actions": list(ACTION_NAMES),
        "limitations": [
            "Synthetic cooperative environment; not real bank data or Bank of Beijing policy.",
            "Engineered structured-state features; no natural-language understanding or RAG.",
            "Only tool names are learned; argument binding and threshold arithmetic are rules.",
            "Group-relative REINFORCE has no PPO clipping/old-policy ratio; not LLM GRPO.",
            "Test shares workflow mechanics; stress holds out fault combinations, not arbitrary banking domains.",
            "Reward uses simulator ground truth at episode end; inference sees messages only.",
            "Same test tasks across seeds; seed standard deviation is not independent-task confidence.",
        ],
    }
    dump(out / "config.json", config)
    tasks = {s: generate_tasks(s, n=n) for s, n in config["data_counts"].items()}
    manifest = {}
    for split, items in tasks.items():
        manifest[split] = {"n": len(items), "sha256_canonical_json": hash_json(items),
                           "task_ids": [t["task_id"] for t in items]}
    source_files = ["bank_agent/env.py", "bank_agent/data.py", "bank_agent/policies.py",
                    "bank_agent/toy_policy.py", "scripts/run_toy_experiment.py"]
    dump(out / "manifest.json", {
        "data": manifest, "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files
        },
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    normal_x, normal_y, normal_meta = expert_dataset(tasks["train"], normal_only=True)
    recover_x, recover_y, recover_meta = expert_dataset(tasks["train"])
    dump(out / "demonstrations.json", {
        "normal_only": {"tasks": len(normal_meta), "transitions": len(normal_x), "task_manifest": normal_meta},
        "with_recovery": {"tasks": len(recover_meta), "transitions": len(recover_x), "task_manifest": recover_meta},
        "equal_sampled_transitions_per_arm_per_seed": args.sft_steps * 128,
        "sampling": "with replacement; fixed optimization and sampled-transition budget",
    })
    final_models = []
    for seed in args.seeds:
        print(f"Training seed={seed}", flush=True)
        initial = TinyPolicy(seed=seed)
        for arm, x, y in [("normal_sft", normal_x, normal_y), ("recovery_sft", recover_x, recover_y)]:
            model = initial.clone()
            log = fit_supervised(model, x, y, seed=seed, steps=args.sft_steps)
            jsonl(out / f"train_seed{seed}_{arm}.jsonl", log)
            model.save(out / f"checkpoint_seed{seed}_{arm}.json")
            final_models.append((seed, arm, model))
            print(f"  {arm}: loss {log[0]['full_training_loss']:.4f} -> {log[-1]['full_training_loss']:.4f}", flush=True)
            if arm == "recovery_sft":
                rl_model = model.clone()
                rl_log = fit_group_reinforce(rl_model, tasks["train"], seed=seed, updates=args.rl_updates)
                jsonl(out / f"train_seed{seed}_recovery_sft_group_reinforce.jsonl", rl_log)
                rl_model.save(out / f"checkpoint_seed{seed}_recovery_sft_group_reinforce.json")
                final_models.append((seed, "recovery_sft_group_reinforce", rl_model))
                print(f"  RL: {sum(r['sampled_episodes'] for r in rl_log)} train episodes", flush=True)
    # Test/stress rollouts happen only after all training is complete.
    all_metrics = []
    for seed, arm, model in final_models:
        for split in ("valid", "test", "stress"):
            print(f"Evaluating seed={seed} arm={arm} split={split}", flush=True)
            all_metrics += evaluate(model, tasks[split], out / f"eval_seed{seed}_{arm}_{split}.jsonl",
                                    seed, arm, split)
    dump(out / "metrics_by_seed_and_group.json", all_metrics)
    with (out / "metrics_by_seed_and_group.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    aggregates = []
    for split, arm, group in sorted({(r["split"], r["arm"], r["group"]) for r in all_metrics}):
        rows = [r for r in all_metrics if (r["split"], r["arm"], r["group"]) == (split, arm, group)]
        agg = {"split": split, "arm": arm, "group": group, "seeds": [r["seed"] for r in rows],
               "n_tasks_per_seed": rows[0]["n"]}
        for key in ("success_rate", "mean_steps", "mean_errors", "false_completion_rate",
                    "truncation_rate", "authorized_success_rate", "legal_task_false_refusal_rate", "unauthorized_safe_handoff_rate"):
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                agg[key + "_mean"] = float(np.mean(values))
                agg[key + "_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregates.append(agg)
    dump(out / "summary.json", {
        "label": "Actual CPU toy-policy results, NOT LLM/bank deployment/official tau-bench results",
        "wall_seconds": time.perf_counter() - started, "aggregates": aggregates,
    })
    print(json.dumps([a for a in aggregates if a["group"] == "overall"], indent=2))
    print(f"Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
