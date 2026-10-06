import copy
import json
from bank_agent.data import generate_tasks, canonical_hash
from bank_agent.env import BankEnv
from bank_agent.policies import rollout, action, decide
from bank_agent.evaluate import aggregate

def customer():
    return next(t for t in generate_tasks("train", 30) if t["role"] == "customer" and t["principal_id"] == t["owner_id"] and not t["faults"])

def test_recovery_reference_on_every_split():
    for split in ("train", "valid", "test", "stress"):
        for task in generate_tasks(split, 100):
            env = rollout(task)
            assert env.result()["success"], (task, env.trace)
            assert env.state["writes"] <= 1

def test_identity_is_not_model_controlled():
    task = customer()
    task["principal_id"] = "another-user"
    env = BankEnv(task)
    _, _, _, _ = env.step(action("get_case", principal_id=task["owner_id"]))
    assert env.latest["result"]["code"] == "INVALID_ARGUMENT"
    env.step(action("get_case"))
    assert env.latest["result"]["code"] == "ACCESS_DENIED"
    assert "amount" not in env.latest["result"]
    assert env.state["writes"] == 0

def test_lost_ack_is_committed_once():
    task = customer()
    task["faults"] = ["lost_ack"]
    env = rollout(task)
    assert env.result()["success"]
    assert env.state["writes"] == 1
    timeout = next(i for i,e in enumerate(env.trace) if e["observation"]["result"].get("code") == "WRITE_TIMEOUT")
    assert env.trace[timeout+1]["action"]["tool"] == "get_status"

def test_idempotency_replays_without_second_write():
    env = BankEnv(customer())
    while not env.idempotency:
        env.step(decide(env.messages))
    written = next(e["action"] for e in env.trace if e["action"]["tool"] == "submit_case")
    before = env.state["writes"]
    env.step(written)
    assert env.latest["result"]["idempotent_replay"]
    assert env.state["writes"] == before == 1

def test_refusal_and_fabricated_completion_are_not_success():
    for tool, args in [("handoff", {"reason":"uncertain"}), ("finish", {"status":"submitted"})]:
        env = BankEnv(customer())
        _, reward, done, info = env.step(action(tool, **args))
        assert done and reward == 0 and not info["success"]

def test_observation_hides_fault_plan_and_ground_truth():
    task = customer()
    task["faults"] = ["lost_ack"]
    obs = BankEnv(task).reset()
    text = json.dumps(obs)
    for hidden in ("lost_ack", "owner_id", "principal_id", "target_status"):
        assert hidden not in text

def test_splits_are_deterministic_and_disjoint():
    all_splits = [generate_tasks(s, 100) for s in ("train","valid","test","stress")]
    assert canonical_hash(all_splits[0]) == canonical_hash(generate_tasks("train",100))
    for i,a in enumerate(all_splits):
        for b in all_splits[i+1:]:
            for key in ("task_id","case_id","owner_id","template_id"):
                assert not ({x[key] for x in a} & {x[key] for x in b})

def test_truncation_kept_and_split_metrics_not_inflated_by_refusal():
    env = BankEnv(customer(), max_steps=1)
    env.step(action("search_policy"))
    row = env.result()
    result = aggregate([row])
    assert row["truncated"] and not row["success"]
    assert result["n_authorized"] == 1 and result["authorized_success_rate"] == 0

def test_unacknowledged_write_cannot_be_guessed_as_success():
    task = customer()
    task["faults"] = ["lost_ack"]
    env = BankEnv(task)
    while not env.state["writes"]:
        env.step(decide(env.messages))
    _, reward, _, info = env.step(action("finish", status="submitted"))
    assert info["unverified_completion"] and not info["success"] and reward == 0

def test_invalid_terminal_arguments_do_not_terminate():
    env = BankEnv(customer())
    env.step(action("finish"))
    assert not env.done and env.latest["result"]["code"] == "INVALID_ARGUMENT"
    env.step(action("handoff"))
    assert not env.done and env.latest["result"]["code"] == "INVALID_ARGUMENT"

def test_public_ids_do_not_include_task_sequence():
    import re
    for t in generate_tasks("train", 100):
        assert re.fullmatch(r"case-[0-9a-f]{16}", t["case_id"])
        assert re.fullmatch(r"request-[0-9a-f]{16}", t["request_id"])


def test_canonical_corpus_hashes_match_published_protocol():
    # Content hashes also catch damaged Unicode during artifact publishing.
    expected = {
        "train": (500, "1fae9b7f0ebf6954be5bc1ffe83c199a52a0a4409ff357587d36c84265a86bd2"),
        "valid": (200, "8f8270bd5293b5be400a64edc04bc61dd97025a5df503f032a4c0a77b961e89b"),
        "test": (200, "624c5b2c3c464a4886599d7aabcc0a9a159f183367a8115cc57d83a7b0118643"),
        "stress": (200, "42950b9fd784911be32f01219bedc555afa9d1f1585b33f41796a436939a1cee"),
    }
    for split, (count, digest) in expected.items():
        assert canonical_hash(generate_tasks(split, count)) == digest
