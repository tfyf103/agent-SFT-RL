"""Chat-completions HTTP adapter; works with a separately running local model server."""
import json
import os
import urllib.request

def parse_action(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("tool"), str):
        raise ValueError("Expected JSON action object")
    if not isinstance(value.get("arguments", {}), dict):
        raise ValueError("arguments must be an object")
    return value

class ChatPolicy:
    def __init__(self, base_url, model, seed=42, temperature=0, timeout=120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.seed, self.temperature, self.timeout = model, seed, temperature, timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
    def __call__(self, messages):
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature, "seed": self.seed,
                           "max_tokens": 256}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("BANK_AGENT_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        for name in self.usage:
            self.usage[name] += data.get("usage", {}).get(name, 0)
        return parse_action(data["choices"][0]["message"]["content"])
