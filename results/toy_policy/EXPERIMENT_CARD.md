# CPU toy-policy experiment card

**Actual locally executed NumPy MLP training. This is not an LLM SFT/RL run, not Slime GRPO, not an official tau-bench score, and not a real-bank deployment.**

## What was learned

The 1,416-parameter MLP (35 public-state features, 32 tanh hidden units, 8 actions) learns only which tool name to call. Inputs come exclusively from observable `env.messages`. Parsing the structured history, binding request IDs and policy versions, selecting the final status string, and comparing the observed amount to the observed policy threshold are deterministic adapters. There is no transformer, pretrained language model, retrieval model or learned argument generation.

The setting is deliberately small and mostly finite-state. Success here does not imply understanding unseen customer language or banking rules.

## Fixed protocol

- Seeds: 17, 29, 43; none selected or discarded.
- Tasks: train 500, validation 200, test 200, stress 200.
- Validation/test/stress were evaluated only after all models had finished training. No checkpoint selection or test-driven parameter search.
- The normal-only SFT arm has 100 train tasks / 478 expert transitions.
- Recovery SFT has all 500 train tasks / 2,827 expert transitions.
- Both SFT arms use 400 updates x 128 sampled transitions = **51,200 action examples** per seed. These are action budgets, not LLM token budgets. Sampling is with replacement.
- Both SFT arms start from identical seed-specific random weights.
- RL starts from recovery SFT, with 100 updates x 4 tasks x 8 sampled episodes = **3,200 train episodes per seed**.
- RL is **group-relative REINFORCE**: terminal success reward 0/1, group-standardized advantage, exact categorical KL to a fixed SFT reference, one on-policy update per batch. There are no old-policy ratios or PPO clipping, so this is not called GRPO.
- Inference for reported metrics is greedy argmax, with a maximum of 16 steps.
- Simulator permission checks are identical across all arms.

## Actual final results

The primary task metric below excludes unauthorized tasks. Test has 164 authorized tasks and 36 unauthorized tasks; stress has 157 authorized tasks and 43 unauthorized tasks. Each seed uses the same task set.

| Arm | Test authorized success | Stress authorized success | Test overall success | Stress overall success |
|---|---:|---:|---:|---:|
| Normal-only SFT | 55.89% +/- 5.63 pp | 0.64% +/- 1.10 pp | 63.83% +/- 4.62 pp | 22.00% +/- 0.87 pp |
| Recovery SFT | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp |
| Recovery SFT + group-relative REINFORCE | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp | 100.00% +/- 0.00 pp |

Values are the mean and sample standard deviation across three seeds, not a confidence interval. The repeated seed runs are not independent sets of 200 tasks.

Recovery SFT solves the small greedy-evaluation environment completely. **The experiment provides no evidence of an additional greedy-success improvement from RL.** Reporting the 100% score as an RL improvement would be incorrect.

The low normal-only result under faults reflects missing recovery demonstrations in a tiny structured-state model. It is not a comparison against a state-of-the-art LLM. All seeds and all fixed splits are reported, including failures.

The RL updates are real: 3/400, 1/400 and 2/400 sampled task groups respectively had nonzero group-relative advantages. SFT was already highly confident, so most groups were all-success and offered no reward-contrast signal. The parameter L2 change from the SFT checkpoint was 0.4347, 0.2010 and 0.4030, respectively. This illustrates the zero-variance-group limitation rather than claiming a useful RL gain.

## Environment correction and rerun

An independent contract review found that public sequential IDs exposed a task index, that a guessed completion after a lost acknowledgement could receive success, and that invalid finish arguments needed explicit error handling. Public IDs are now opaque hashes; success requires an observed successful write acknowledgement or a status readback matching the reported final status; malformed finish arguments return INVALID_ARGUMENT, while hitting the step limit is recorded as max_steps.

All three seeds were rerun from initialization with the **identical model, hyperparameters, seeds, data counts and evaluation protocol** after these corrections. The files in this directory replace the earlier provisional run. No parameters or policies were selected based on held-out results. The changed environment also allows a model to continue after an invalid finish argument, which changes some normal-only SFT outcomes. The current table and raw files are the corrected results.

## Audit trail

- `config.json`: full fixed hyperparameters, feature definitions and limitations.
- `manifest.json`: source SHA-256, data SHA-256, runtime and task IDs.
- `demonstrations.json`: expert task counts and training-action budget.
- `train_seed*_*.jsonl`: SFT losses and RL per-update reward/gradient/KL records.
- `checkpoint_seed*_*.json`: all nine readable model checkpoints.
- `eval_seed*_*_*.jsonl`: every evaluated task result and complete tool trace.
- `metrics_by_seed_and_group.json/csv`: outcomes broken down by role, authorization, fault combination and synthetic wording family.
- `summary.json`: seed aggregates and measured wall time.
- `training_audit.json`: loss reduction and RL update checks.

All five recorded source hashes and all four regenerated data hashes were checked against the run manifest after execution.

## Verification and reproduction

The eight dedicated unit tests passed: observable-state isolation; deterministic public argument binding; SFT gradient finite differences; REINFORCE + exact-KL gradient finite differences; positive-advantage probability increase; zero reward-variance signal; actual expert-state SFT loss reduction; and training API rejection of held-out tasks.

```bash
python -m unittest discover -s tests -p test_toy_policy.py -v
python scripts/run_toy_experiment.py
```

The original run took about 13.6 seconds on the available Windows CPU Python/NumPy runtime. It did not use ChatGPT cloud GPUs. Timings are machine-dependent.

## Limits on interpretation

Test and stress share the same workflow semantics and tool definitions with training. Stress holds out combinations of injected faults; it is not an unseen banking domain. Different task wording is ignored by this feature-based policy, so wording splits cannot establish natural-language generalization. Authorization is enforced by the backend, and zero unauthorized writes must not be attributed to model safety training. The 100% result reveals the environment's small scope and saturation; it cannot support claims of production reliability, customer-service quality, Bank of Beijing adoption, LLM fine-tuning, or official benchmark reproduction.

