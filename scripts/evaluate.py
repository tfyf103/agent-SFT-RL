import argparse
import json
from bank_agent.env import load_tasks
from bank_agent.evaluate import suite, evaluate
from bank_agent.llm import ChatPolicy
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0)
    args = p.parse_args()
    tasks = load_tasks(args.tasks)
    if args.base_url:
        if not args.model:
            p.error("--model is required for model evaluation")
        policy = ChatPolicy(args.base_url, args.model, args.seed, args.temperature)
        result = evaluate(tasks, policy, args.output, "llm:" + args.model)
        result["provider_reported_token_usage"] = policy.usage
        from pathlib import Path
        (Path(args.output)/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = suite(tasks, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
