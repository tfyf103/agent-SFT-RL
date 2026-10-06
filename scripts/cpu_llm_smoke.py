"""A real 135M CPU SFT/checkpoint/resume smoke, not a bank-quality experiment."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    root = Path("artifacts/llm-smoke")
    root.mkdir(parents=True, exist_ok=True)
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = HfApi().model_info(model_id).sha
    records = []
    for i in range(8):
        records.append({"messages": [
            {"role":"system","content":"Return one JSON tool action. This is a synthetic exercise."},
            {"role":"user","content":f"Read the current policy for synthetic case {i}."},
            {"role":"assistant","content":'{"tool":"search_policy","arguments":{"query":"case procedure"}}'}
        ], "metadata":{"split":"train"}})
    data_path = root/"short-smoke.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in records)+"\n", encoding="utf-8")

    def fingerprint(folder, rev=None):
        m = AutoModelForCausalLM.from_pretrained(folder, revision=rev, torch_dtype=torch.float32)
        h = hashlib.sha256()
        for name, value in m.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        result = h.hexdigest()
        del m
        gc.collect()
        return result

    before = fingerprint(model_id, revision)
    common = [sys.executable,"scripts/train_sft.py","--model",model_id,"--revision",revision,
              "--data",str(data_path),"--output",str(root/"sft"),"--device","cpu",
              "--dtype","float32","--max-samples","8","--max-length","256",
              "--gradient-accumulation","1"]
    subprocess.run(common+["--max-steps","2"],check=True)
    cp2 = root/"sft"/"step-000002"
    after2 = fingerprint(str(cp2/"model"))
    if before == after2:
        raise RuntimeError("SFT did not change weights")
    subprocess.run(common+["--resume",str(cp2),"--max-steps","3"],check=True)
    cp3 = root/"sft"/"step-000003"
    after3 = fingerprint(str(cp3/"model"))
    if after2 == after3:
        raise RuntimeError("Resumed SFT did not change weights")
    saved = torch.load(cp3/"training_state.pt",map_location="cpu",weights_only=False)
    assert saved["state"]["optimizer_steps"] == 3

    result = {"scope":"135M CPU SFT pipeline smoke only; NOT bank task accuracy",
              "model":model_id,"revision":revision,"device":"cpu","optimizer_steps":3,
              "base_state_sha256":before,"step2_state_sha256":after2,"step3_state_sha256":after3,
              "weights_changed":True,"checkpoint_reloaded":True,"resume_verified":True,
              "bank_evaluation_performed":False,"grpo_training_performed":False}
    (root/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
