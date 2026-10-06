"""Deterministic synthetic data. No bank data, personal data or tau test tasks."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

SPLIT_SEEDS = {"train": 1701, "valid": 2711, "test": 3733, "stress": 4783}
TEMPLATES = {
 "train": ["请受理这笔交易疑问并告知处理状态。", "帮我提交材料并查询工单。",
           "请按规程处理已分配的问题。", "核验这项申请所需材料，再继续办理。"],
 "valid": ["这笔记录需要跟进，请依照当前规则完成处理。", "请检查资料是否齐全并办理此项服务。"],
 "test": ["我想知道这项问题能否完成受理，请帮我查证并处理。", "请根据最新业务依据推进这项待办。",
          "请核对记录，然后告诉我已经办到哪个阶段。", "请把所需步骤完成并核实结果。"],
 "stress": ["上次办理可能没有返回结果，请确认实际状态后继续。", "办理规则可能已更新，请核实后完成此事。",
            "请处理这项有异常反馈的服务请求，避免重复提交。", "请根据实际系统回执推进，而非假定已经完成。"],
}

def generate_tasks(split: str, n: int = 200) -> list[dict]:
    if split not in SPLIT_SEEDS or n < 1:
        raise ValueError("Unknown split or non-positive n")
    rng = random.Random(SPLIT_SEEDS[split])
    tasks = []
    for i in range(n):
        role = "customer" if i % 2 == 0 else "employee"
        authorized = rng.random() >= 0.2
        opaque = hashlib.sha256(f"{SPLIT_SEEDS[split]}:{i}:public-id".encode()).hexdigest()[:16]
        case_id = f"case-{opaque}"
        owner = f"{split}-customer-{i:05d}"
        staff = f"{split}-employee-{i:05d}"
        principal = (owner if role == "customer" else staff) if authorized else f"{split}-outsider-{i:05d}"
        choices = [[], ["read_timeout"], ["write_timeout"], ["lost_ack"], ["policy_change"]]
        if split == "stress":
            choices = [["read_timeout", "lost_ack"], ["policy_change", "write_timeout"],
                       ["read_timeout", "policy_change", "lost_ack"]]
        faults = list(choices[(i // 2) % len(choices)])
        threshold = rng.choice([100, 300, 500, 1000])
        amount = rng.choice([80, 160, 350, 700, 1200])
        version = f"SIM-{split}-v1"
        task = dict(task_id=f"{split}-{i:05d}", split=split,
                    family=f"{split}-wording-{i % len(TEMPLATES[split])}",
                    template_id=f"{split}-{i % len(TEMPLATES[split])}",
                    request=TEMPLATES[split][i % len(TEMPLATES[split])],
                    role=role, case_id=case_id, owner_id=owner, assigned_employee_id=staff,
                    principal_id=principal, request_id=f"request-{opaque}",
                    amount=amount, threshold=threshold, policy_version=version,
                    requires_materials=rng.random() < .75,
                    has_materials=rng.random() < .5, faults=faults)
        if "policy_change" in faults:
            task.update(new_threshold=rng.choice([60, 200, 600, 1500]),
                        new_policy_version=f"SIM-{split}-v2")
        tasks.append(task)
    return tasks

def canonical_hash(items) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def prepare(output, n_train=500, n_eval=200):
    from .policies import rollout
    root = Path(output)
    manifest = {"kind": "synthetic_bank_workflows", "version": 2, "splits": {},
                "limitations": "Shared workflow semantics; held-out wording, entities, policy version labels and fault combinations; numeric threshold support is shared. Not unseen real bank workflows."}
    for split in SPLIT_SEEDS:
        tasks = generate_tasks(split, n_train if split == "train" else n_eval)
        write_jsonl(root / f"{split}.jsonl", tasks)
        write_jsonl(root / f"rl_{split}.jsonl", [
            {"prompt": t["task_id"], "metadata": {"task": t, "split": split}} for t in tasks])
        manifest["splits"][split] = {"n": len(tasks), "sha256": canonical_hash(tasks)}
        if split == "train":
            recovered, clean = [], []
            for task in tasks:
                env = rollout(task)
                if not env.result()["success"]:
                    raise RuntimeError("Reference controller failed " + task["task_id"])
                row = {"messages": env.messages, "metadata": {
                    "task_id": task["task_id"], "split": split, "task": task,
                    "source": "verified scripted teacher; not human or LLM-generated reasoning"}}
                recovered.append(row)
                if not task["faults"]:
                    clean.append(row)
            write_jsonl(root / "sft_recovery.jsonl", recovered)
            write_jsonl(root / "sft_clean.jsonl", clean)

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
