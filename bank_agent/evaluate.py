"""Paired synthetic evaluation, trace retention and transparent metric denominators."""
from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
from .data import canonical_hash, write_jsonl
from .env import BankEnv
from .policies import decide

def wilson(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.96
    d = 1 + z*z/total
    mid = (p + z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0, mid-half), min(1, mid+half)]

def aggregate(rows):
    authorized = [r for r in rows if r["authorized"]]
    denied = [r for r in rows if not r["authorized"]]
    fault = [r for r in authorized if r["faults"]]
    def rate(group, key="success"):
        return sum(bool(r[key]) for r in group)/len(group) if group else None
    n_ok = sum(r["success"] for r in authorized)
    return {"n": len(rows), "n_authorized": len(authorized), "n_denied": len(denied),
            "authorized_success_rate": rate(authorized),
            "authorized_success_wilson95": wilson(n_ok, len(authorized)),
            "unauthorized_correct_handling_rate": rate(denied),
            "fault_task_success_rate": rate(fault),
            "authorized_false_refusal_rate": rate(authorized, "false_refusal"),
            "false_completion_rate": rate(authorized, "false_completion"),
            "truncation_rate": rate(rows, "truncated"),
            "mean_steps_all": sum(r["steps"] for r in rows)/len(rows) if rows else None,
            "mean_steps_successful_authorized": (sum(r["steps"] for r in authorized if r["success"])/n_ok) if n_ok else None,
            "write_count_exceeds_one": sum(r["writes"] > 1 for r in rows),
            "note": "Task-level synthetic metrics. Scripted runs have no sampling; do not interpret repeat runs as pass^k."}

def evaluate(tasks, policy, output, label, max_steps=16):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, traces = [], []
    for task in tasks:
        env = BankEnv(task, max_steps=max_steps)
        while not env.done:
            try:
                chosen = policy(env.messages)
            except Exception as error:
                # Provider/decoding failures remain in denominator and raw logs.
                chosen = {"tool": "__provider_error__", "arguments": {}}
                env.provider_error = type(error).__name__
            env.step(chosen)
        row = env.result()
        if hasattr(env, "provider_error"):
            row["provider_error"] = env.provider_error
        rows.append(row)
        traces.append({"task_id": task["task_id"], "result": row, "trace": env.trace})
    summary = aggregate(rows)
    summary.update(label=label, dataset_sha256=canonical_hash(tasks),
                   runtime={"python": platform.python_version(), "platform": platform.platform()})
    write_jsonl(output / "episodes.jsonl", rows)
    write_jsonl(output / "traces.jsonl", traces)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

def suite(tasks, output):
    results = {}
    for mode in ("open_loop", "no_clarification", "no_readback", "recovery"):
        results[mode] = evaluate(tasks, lambda m, mode=mode: decide(m, mode), Path(output)/mode,
                                 label=f"scripted_{mode}")
    report = {"execution": "actual_cpu_scripted_controls", "llm_training_executed": False,
              "results": results}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/"comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
