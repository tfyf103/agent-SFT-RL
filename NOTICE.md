# Sources and attribution

This project is an independently implemented small synthetic banking workflow
laboratory. It is not affiliated with Bank of Beijing and has no real banking
API, customer database or internal policy document.

- Sierra Research tau-bench / tau-Knowledge: MIT. Environment/evaluation design reference and separately runnable external benchmark.
  https://github.com/sierra-research/tau2-bench
- THUDM Slime: Apache-2.0. Optional training framework and adapter protocol reference.
  https://github.com/THUDM/slime
- Hugging Face Transformers / SmolLM2: Apache-2.0. Optional model training backend and CPU smoke-test model.
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

Upstream repositories are not vendored or rebranded. Pin and attribution details
are in upstream.lock.json and docs/UPSTREAM_REPRODUCTION.md. Any upstream files
downloaded by reproduction scripts retain their original licenses. Our generated
policy texts, identities, cases and numerical business thresholds are synthetic.
They are not copies of a bank's real operating procedures.

Public upstream benchmark scores are never copied into our measured results.
