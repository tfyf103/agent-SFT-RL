"""Observable-history scripted controls; these are NOT trained language models."""
import json

def context(messages):
    first = json.loads(messages[1]["content"])
    policy, case, last, status = None, None, None, None
    for message in messages:
        text = message.get("content", "")
        if message["role"] != "user" or not text.startswith("[ENVIRONMENT] "):
            continue
        event = json.loads(text[len("[ENVIRONMENT] "):])
        last = event
        result = event["result"]
        if result.get("ok"):
            if event["tool"] == "search_policy":
                policy = result
            elif event["tool"] == "get_case":
                case = result
            elif event["tool"] == "collect_materials" and case:
                case = dict(case, materials=True)
            elif event["tool"] == "get_status":
                status = result
        elif result.get("code") == "STALE_POLICY":
            policy = None
    return first, policy, case, last, status

def action(tool, **arguments):
    return {"tool": tool, "arguments": arguments}

def decide(messages, mode="recovery"):
    first, policy, case, last, status = context(messages)
    if last and last["result"].get("code") == "ACCESS_DENIED":
        return action("handoff", reason="access_denied")
    # The open-loop control uses the same tools and permission gateway.
    if mode == "open_loop" and last and not last["result"].get("ok"):
        return action("finish", status="submitted" if first["role"] == "customer" else "resolved")
    if last and last["result"].get("code") == "WRITE_TIMEOUT":
        if mode == "no_readback":
            return action("finish", status="submitted" if first["role"] == "customer" else "resolved")
        return action("get_status")
    if status and status["writes"] == 1:
        return action("finish", status=status["status"])
    if not policy:
        return action("search_policy", query="applicable case procedure")
    if not case:
        return action("get_case")
    if policy["requires_materials"] and not case["materials"]:
        if mode == "no_clarification":
            return action("handoff", reason="missing_materials")
        return action("collect_materials")
    if last and last["tool"] in ("submit_case", "resolve_case") and last["result"].get("ok"):
        if mode == "no_readback":
            return action("finish", status=last["result"]["status"])
        return action("get_status")
    args = {"policy_version": policy["version"], "idempotency_key": first["request_id"]}
    if first["role"] == "customer":
        return action("submit_case", **args)
    resolution = "escalated" if case["amount"] > policy["escalation_threshold"] else "resolved"
    return action("resolve_case", resolution=resolution, **args)

def rollout(task, mode="recovery", max_steps=16):
    from .env import BankEnv
    env = BankEnv(task, max_steps=max_steps)
    while not env.done:
        env.step(decide(env.messages, mode))
    return env
