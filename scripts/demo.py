"""Print a full, deterministic recovery walkthrough without a model or API key."""
import argparse
import json
from bank_agent.data import generate_tasks
from bank_agent.policies import rollout
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["customer","employee"], default="customer")
    p.add_argument("--mode", choices=["open_loop","no_clarification","no_readback","recovery"], default="recovery")
    args = p.parse_args()
    task = next(t for t in generate_tasks("stress",200)
                if t["role"] == args.role
                and t["principal_id"] == (t["owner_id"] if args.role == "customer" else t["assigned_employee_id"])
                and "lost_ack" in t["faults"])
    env = rollout(task,args.mode)
    print("SYNTHETIC SCRIPTED DEMO — not an LLM or real bank. Hidden fault plan is not passed to policy.")
    print("Request:", task["request"])
    for event in env.trace:
        print(json.dumps(event,ensure_ascii=False,indent=2))
    print("Final result:",json.dumps(env.result(),ensure_ascii=False,indent=2))
