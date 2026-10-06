"""CPU-only, observable-state MLP experiment; this is NOT a language model.

The learned component selects one of eight tool names. JSON parsing, public-state
features, request-id binding, policy-version binding and numeric threshold
comparison are deterministic adapters. Thus results do not measure language
understanding, JSON generation, RAG, learned argument generation or LLM GRPO.
The RL algorithm is on-policy group-relative REINFORCE, with exact categorical
KL to a frozen SFT reference. It has no old-policy ratios or PPO clipping.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .env import BankEnv
from .policies import action, context, decide

ACTION_NAMES = (
    "search_policy", "get_case", "collect_materials", "get_status",
    "submit_case", "resolve_case", "handoff", "finish",
)
ERROR_CODES = (
    "ACCESS_DENIED", "READ_TIMEOUT", "WRITE_TIMEOUT", "STALE_POLICY",
    "MISSING_MATERIALS", "READ_REQUIRED", "ROLE_DENIED", "WRONG_RESOLUTION",
    "INVALID_REQUEST_KEY", "INVALID_ARGUMENT", "OTHER",
)
STATE_NAMES = ("draft", "assigned", "submitted", "resolved", "escalated")
FEATURE_NAMES = (
    "bias", "customer", "policy_observed", "case_observed", "materials_present",
    "requires_materials", "status_observed", "one_write_verified",
    "last_response_ok", "last_response_failed", "turn_fraction",
) + tuple("last_tool:" + n for n in ACTION_NAMES) + tuple(
    "last_error:" + n for n in ERROR_CODES
) + tuple("observed_status:" + n for n in STATE_NAMES)


def features(messages):
    """Only inspect observable message history; never accepts a task/environment."""
    first, policy, case, last, status = context(messages)
    result = last["result"] if last else {}
    tool = last["tool"] if last else None
    code = result.get("code")
    if code and code not in ERROR_CODES:
        code = "OTHER"
    observed_status = (status or case or {}).get("status")
    turns = sum(m.get("role") == "assistant" for m in messages)
    values = [
        1.0, first["role"] == "customer", policy is not None, case is not None,
        bool(case and case.get("materials")), bool(policy and policy.get("requires_materials")),
        status is not None, bool(status and status.get("writes") == 1),
        bool(last and result.get("ok")), bool(last and not result.get("ok")),
        min(turns / 16.0, 1.0),
    ]
    values += [tool == name for name in ACTION_NAMES]
    values += [code == name for name in ERROR_CODES]
    values += [observed_status == name for name in STATE_NAMES]
    return np.asarray(values, dtype=np.float64)


def bind_action(action_index, messages):
    """Bind arguments deterministically from public observations, NOT learned."""
    first, policy, case, last, status = context(messages)
    name = ACTION_NAMES[int(action_index)]
    if name == "search_policy":
        return action(name, query="applicable case procedure")
    if name == "handoff":
        return action(name, reason="access_denied")
    if name == "finish":
        current = (status or {}).get("status")
        if current is None and last and last["result"].get("ok"):
            current = last["result"].get("status")
        return action(name, status=current or (case or {}).get("status", "unknown"))
    if name in ("submit_case", "resolve_case"):
        args = {
            "policy_version": (policy or {}).get("version", "unknown"),
            "idempotency_key": first["request_id"],
        }
        if name == "resolve_case":
            amount = (case or {}).get("amount", 0)
            threshold = (policy or {}).get("escalation_threshold", float("inf"))
            args["resolution"] = "escalated" if amount > threshold else "resolved"
        return action(name, **args)
    return action(name)


class TinyPolicy:
    """One hidden tanh layer; no transformer, tokenizer or pretrained weights."""
    def __init__(self, seed=17, hidden=32):
        rng = np.random.default_rng(seed)
        self.hidden = int(hidden)
        self.params = {
            "w1": rng.normal(0, 1 / np.sqrt(len(FEATURE_NAMES)), (len(FEATURE_NAMES), hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, len(ACTION_NAMES))),
            "b2": np.zeros(len(ACTION_NAMES)),
        }

    def clone(self):
        return copy.deepcopy(self)

    def forward(self, x):
        x = np.atleast_2d(x)
        h = np.tanh(x @ self.params["w1"] + self.params["b1"])
        logits = h @ self.params["w2"] + self.params["b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(shifted)
        p /= p.sum(axis=1, keepdims=True)
        return p, (x, h)

    def probabilities(self, x):
        return self.forward(x)[0]

    def backward(self, cache, d_logits):
        x, h = cache
        d_hidden = (d_logits @ self.params["w2"].T) * (1 - h * h)
        return {
            "w1": x.T @ d_hidden, "b1": d_hidden.sum(axis=0),
            "w2": h.T @ d_logits, "b2": d_logits.sum(axis=0),
        }

    def supervised_loss_grad(self, x, labels):
        p, cache = self.forward(x)
        labels = np.asarray(labels, dtype=np.int64)
        loss = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        dz = p.copy()
        dz[np.arange(len(labels)), labels] -= 1
        return float(loss), self.backward(cache, dz / len(labels))

    def reinforce_loss_grad(self, x, chosen, advantages, reference, beta, divisor):
        """-A*log pi + beta*KL(pi||reference), summed per rollout then averaged.

        Each transition receives its rollout's group-standardized terminal advantage.
        The KL is exact over the eight actions at sampled states. It is a surrogate
        regularizer: no gradient through the state distribution is estimated for KL.
        """
        p, cache = self.forward(x)
        q = reference.probabilities(x)
        chosen = np.asarray(chosen, dtype=np.int64)
        advantages = np.asarray(advantages)
        log_ratio = np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(q, 1e-12, 1))
        kl = (p * log_ratio).sum(axis=1)
        loss = (-advantages * np.log(np.clip(p[np.arange(len(chosen)), chosen], 1e-12, 1))
                + beta * kl).sum() / divisor
        dz = p.copy()
        dz[np.arange(len(chosen)), chosen] -= 1
        dz *= advantages[:, None]
        dz += beta * p * (log_ratio - kl[:, None])
        return float(loss), self.backward(cache, dz / divisor), float(kl.mean())

    def save(self, path):
        payload = {
            "type": "cpu_tiny_mlp_tool_selector_not_llm",
            "hidden": self.hidden, "feature_names": FEATURE_NAMES,
            "action_names": ACTION_NAMES,
            "parameters": {k: v.tolist() for k, v in self.params.items()},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["feature_names"] != list(FEATURE_NAMES) or payload["action_names"] != list(ACTION_NAMES):
            raise ValueError("Checkpoint feature/action schema mismatch.")
        model = cls(hidden=payload["hidden"])
        model.params = {k: np.asarray(v, dtype=np.float64) for k, v in payload["parameters"].items()}
        return model


class Adam:
    def __init__(self, params, learning_rate):
        self.lr = learning_rate
        self.step = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update(self, params, grads, max_norm=1.0):
        self.step += 1
        norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
        scale = min(1.0, max_norm / (norm + 1e-12))
        for key in params:
            g = grads[key] * scale
            self.m[key] = 0.9 * self.m[key] + 0.1 * g
            self.v[key] = 0.999 * self.v[key] + 0.001 * g * g
            m_hat = self.m[key] / (1 - 0.9 ** self.step)
            v_hat = self.v[key] / (1 - 0.999 ** self.step)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return norm


def group_advantages(rewards):
    rewards = np.asarray(rewards, dtype=np.float64)
    std = float(rewards.std())
    if std < 1e-12:
        return np.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + 1e-8)


def expert_dataset(tasks, normal_only=False):
    """Fault metadata is used only for a TRAIN-data ablation, never as a feature."""
    xs, ys, metadata = [], [], []
    for task in tasks:
        if normal_only and task["faults"]:
            continue
        if task["split"] != "train":
            raise ValueError("Expert demonstrations must come from train tasks only.")
        env = BankEnv(task)
        episode_x, episode_y = [], []
        while not env.done:
            answer = decide(env.messages)
            episode_x.append(features(env.messages))
            episode_y.append(ACTION_NAMES.index(answer["tool"]))
            env.step(answer)
        if not env.result()["success"]:
            raise ValueError("Expert failed training task: " + task["task_id"])
        xs.extend(episode_x)
        ys.extend(episode_y)
        metadata.append({"task_id": task["task_id"], "transitions": len(episode_x)})
    if not xs:
        raise ValueError("No expert training examples.")
    return np.asarray(xs), np.asarray(ys, dtype=np.int64), metadata


def fit_supervised(model, x, labels, seed=17, steps=400, batch_size=128, learning_rate=0.01):
    """Both SFT arms use the same number of sampled transitions and updates."""
    rng = np.random.default_rng(seed)
    optim = Adam(model.params, learning_rate)
    initial_loss = model.supervised_loss_grad(x, labels)[0]
    log = [{"step": 0, "full_training_loss": initial_loss}]
    for step in range(1, steps + 1):
        indices = rng.integers(len(x), size=batch_size)
        loss, grad = model.supervised_loss_grad(x[indices], labels[indices])
        norm = optim.update(model.params, grad)
        if step % 25 == 0 or step == steps:
            full_loss = model.supervised_loss_grad(x, labels)[0]
            accuracy = np.mean(model.probabilities(x).argmax(axis=1) == labels)
            log.append({"step": step, "batch_loss": loss, "full_training_loss": full_loss,
                        "training_action_accuracy": float(accuracy), "gradient_norm": norm})
    if not log[-1]["full_training_loss"] < initial_loss:
        raise RuntimeError("SFT did not reduce its training loss.")
    return log


def policy_rollout(model, task, rng=None, max_steps=16):
    env = BankEnv(task, max_steps=max_steps)
    xs, chosen = [], []
    while not env.done:
        x = features(env.messages)
        probabilities = model.probabilities(x)[0]
        index = int(probabilities.argmax() if rng is None else rng.choice(len(ACTION_NAMES), p=probabilities))
        xs.append(x)
        chosen.append(index)
        env.step(bind_action(index, env.messages))
    return env, np.asarray(xs), np.asarray(chosen, dtype=np.int64)


def fit_group_reinforce(model, tasks, seed=17, updates=100, groups_per_update=4,
                        group_size=8, learning_rate=0.001, beta=0.02):
    """On-policy grouped REINFORCE, not the clipped GRPO implementation in Slime."""
    if any(task["split"] != "train" for task in tasks):
        raise ValueError("RL may only sample train tasks.")
    if group_size < 2:
        raise ValueError("Group-relative advantages need at least two rollouts.")
    rng = np.random.default_rng(seed)
    reference = model.clone()
    optim = Adam(model.params, learning_rate)
    log = []
    for update in range(1, updates + 1):
        all_x, all_y, all_adv = [], [], []
        group_rewards, nonzero_groups, episode_steps = [], 0, []
        for _ in range(groups_per_update):
            task = tasks[int(rng.integers(len(tasks)))]
            episodes, rewards = [], []
            for _ in range(group_size):
                env, x, y = policy_rollout(model, task, rng)
                episodes.append((x, y))
                rewards.append(float(env.result()["success"]))
                episode_steps.append(env.steps)
            advantages = group_advantages(rewards)
            nonzero_groups += int(np.any(advantages))
            group_rewards.extend(rewards)
            for (x, y), advantage in zip(episodes, advantages):
                all_x.extend(x)
                all_y.extend(y)
                all_adv.extend([advantage] * len(y))
        loss, grad, mean_kl = model.reinforce_loss_grad(
            np.asarray(all_x), all_y, all_adv, reference, beta,
            divisor=groups_per_update * group_size,
        )
        norm = optim.update(model.params, grad)
        log.append({
            "update": update, "sampled_train_success_rate": float(np.mean(group_rewards)),
            "sampled_train_mean_steps": float(np.mean(episode_steps)),
            "nonzero_advantage_groups": nonzero_groups, "groups": groups_per_update,
            "sampled_state_mean_kl": mean_kl, "surrogate_loss": loss, "gradient_norm": norm,
            "sampled_episodes": groups_per_update * group_size,
        })
    return log
