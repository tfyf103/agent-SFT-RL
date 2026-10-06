"""CPU helper checks. No pretrained-model or GPU result is implied."""
from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bank_agent.training import (
    ExampleTooLong, Trajectory, Turn, assert_sft_disjoint, chat_ids, clipped_objective,
    completion_logprobs, grpo_update, group_advantages, k3_kl,
    load_sft_examples, parse_action, record_partition, sft_example, validate_resume, validate_training_tasks,
)


class StableTokenizer:
    """A toy character tokenizer for proving span logic, not an LLM tokenizer."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = "".join(
            f"<{m['role']}>{m['content']}~" for m in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


class RewritingTokenizer(StableTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return [len(messages)] + super().apply_chat_template(
            messages, tokenize, add_generation_prompt
        )


class TrainingHelpersTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = StableTokenizer()
        self.messages = [
            {"role": "system", "content": "Use JSON actions."},
            {"role": "user", "content": "Help with a case."},
            {"role": "assistant", "content": '{"tool":"get_case","arguments":{}}'},
            {"role": "user", "content": "[ENVIRONMENT] missing_materials"},
            {"role": "assistant", "content": '{"tool":"collect_materials","arguments":{}}'},
        ]

    def test_assistant_only_spans_include_turn_end_but_not_role_header(self):
        example = sft_example(self.tokenizer, self.messages, 4096)
        decoded = []
        for start, end in example["assistant_spans"]:
            decoded.append("".join(map(chr, example["input_ids"][start:end])))
            self.assertEqual(example["labels"][start:end], example["input_ids"][start:end])
        self.assertEqual(decoded, [self.messages[2]["content"] + "~", self.messages[4]["content"] + "~"])
        supervised = {i for start, end in example["assistant_spans"] for i in range(start, end)}
        for index, label in enumerate(example["labels"]):
            self.assertEqual(label != -100, index in supervised)

    def test_environment_observation_is_not_supervised(self):
        example = sft_example(self.tokenizer, self.messages, 4096)
        target = "".join(chr(x) for x in example["labels"] if x != -100)
        self.assertNotIn("missing_materials", target)
        self.assertNotIn("Use JSON actions", target)

    def test_unstable_template_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix-stable"):
            sft_example(RewritingTokenizer(), self.messages, 4096)

    def test_overlong_conversation_rejected_whole(self):
        with self.assertRaises(ExampleTooLong):
            sft_example(self.tokenizer, self.messages, 30)

    def test_observation_must_use_explicit_user_role(self):
        with self.assertRaisesRegex(ValueError, "user-role"):
            chat_ids(self.tokenizer, [{"role": "tool", "content": "result"}])

    def test_no_assistant_rejected(self):
        with self.assertRaisesRegex(ValueError, "No trainable"):
            sft_example(self.tokenizer, self.messages[:2], 4096)

    def test_assistant_without_context_rejected(self):
        with self.assertRaisesRegex(ValueError, "preceding"):
            sft_example(self.tokenizer, [self.messages[2]], 4096)

    def test_valid_action_and_malformed_action(self):
        valid = {"tool": "finish", "arguments": {"status": "submitted"}}
        self.assertEqual(parse_action('{"tool":"finish","arguments":{"status":"submitted"}}'), valid)
        for invalid in ["not json", "[]", '{"tool":"finish"}',
                        '{"tool":"finish","arguments":[]}', '{"tool":"finish","arguments":{},"extra":1}']:
            self.assertEqual(parse_action(invalid)["tool"], "__invalid_action__")

    def test_group_advantages_centered_and_scaled(self):
        advantages = group_advantages([0.0, 1.0])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertAlmostEqual(advantages[1], 1.0, places=5)
        self.assertAlmostEqual(advantages[0], -1.0, places=5)
        self.assertEqual(group_advantages([1.0] * 4), [0.0] * 4)

    def test_invalid_reward_groups_fail(self):
        for rewards in [[1.0], [math.inf, 0.0], [math.nan, 1.0]]:
            with self.assertRaises(ValueError):
                group_advantages(rewards)

    def test_ppo_clips_both_advantage_signs(self):
        self.assertAlmostEqual(clipped_objective(1.5, 2.0, 0.2), 2.4)
        self.assertAlmostEqual(clipped_objective(0.5, -2.0, 0.2), -1.6)
        self.assertAlmostEqual(clipped_objective(1.5, -2.0, 0.2), -3.0)

    def test_kl_estimator_nonnegative_and_zero_at_reference(self):
        self.assertEqual(k3_kl(-1.0, -1.0), 0.0)
        for delta in [-4, -1, 0.1, 2]:
            self.assertGreaterEqual(k3_kl(-5, -5 + delta), 0.0)

    def test_resume_cannot_silently_change_reference(self):
        with self.assertRaisesRegex(ValueError, "model"):
            validate_resume({"model": "B"}, {"model": "A"}, {"output", "resume"})
        validate_resume(
            {"model": "A", "output": "new"}, {"model": "A", "output": "old"}, {"output"}
        )

    def test_sft_loader_reports_rejection_counts(self):
        import json
        short = self.messages[:3]
        long = [dict(message) for message in short]
        long[1]["content"] = "long" * 1000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps({"messages": x}) for x in [short, long]), encoding="utf-8")
            examples, counts = load_sft_examples(str(path), self.tokenizer, 512)
        self.assertEqual(len(examples), 1)
        self.assertEqual(counts, {"total": 2, "used": 1, "overlength_rejected": 1,
                                  "unlabeled_split": 2, "split_counts": {}})

    def test_grpo_rejects_heldout_and_duplicate_tasks(self):
        validate_training_tasks([{"task_id": "train-a", "split": "train"}])
        for tasks in [
            [{"task_id": "test-a", "split": "test"}],
            [{"task_id": "a"}],
            [{"task_id": "same", "split": "train"}] * 2,
        ]:
            with self.assertRaises(ValueError):
                validate_training_tasks(tasks)

    def test_metadata_cannot_disguise_heldout_split(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            record_partition({"metadata": {"split": "train", "task": {"split": "test"}}})

    def test_sft_rejects_test_records_even_when_overlong(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({
                "messages": self.messages, "metadata": {"split": "test"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Disallowed split=test"):
                load_sft_examples(str(path), self.tokenizer, 5)

    def test_validation_accepts_valid_but_not_train(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({
                "messages": self.messages, "metadata": {"split": "valid"},
            }), encoding="utf-8")
            examples, counts = load_sft_examples(
                str(path), self.tokenizer, 4096, allowed_splits=("valid", "validation")
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(counts["split_counts"], {"valid": 1})
            with self.assertRaisesRegex(ValueError, "Disallowed"):
                load_sft_examples(str(path), self.tokenizer, 4096)

    def test_train_validation_disjointness_checks_ids_and_content(self):
        with self.assertRaisesRegex(ValueError, "task_id"):
            assert_sft_disjoint(
                [{"task_id": "same", "messages_sha256": "a"}],
                [{"task_id": "same", "messages_sha256": "b"}],
            )
        with self.assertRaisesRegex(ValueError, "messages_sha256"):
            assert_sft_disjoint(
                [{"task_id": None, "messages_sha256": "same"}],
                [{"task_id": None, "messages_sha256": "same"}],
            )
        assert_sft_disjoint(
            [{"task_id": "train", "messages_sha256": "a"}],
            [{"task_id": "valid", "messages_sha256": "b"}],
        )

    def test_empty_trajectory_still_exists(self):
        empty = Trajectory([], 0.0, False, "context_limit", {"success": False})
        self.assertEqual(empty.tokens, 0)
        self.assertEqual(len(group_advantages([1.0, empty.reward])), 2)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Optional torch is not installed")
class TorchObjectiveTests(unittest.TestCase):
    """Actual gradient tests with a four-token bigram model, not a trained LLM."""

    def model(self):
        import torch

        class Bigram(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.zeros(4, 4))

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return SimpleNamespace(logits=self.table[input_ids])
        return Bigram()

    def test_completion_alignment_excludes_prompt_predictions(self):
        import torch
        model = self.model()
        with torch.no_grad():
            model.table[1, 2] = 3.0
            model.table[2, 3] = 2.0
        actual = completion_logprobs(model, [0, 1], [2, 3])
        expected = torch.stack([
            model.table[1].log_softmax(-1)[2],
            model.table[2].log_softmax(-1)[3],
        ])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2,))

    def test_empty_failure_keeps_fixed_group_denominator(self):
        import torch
        policy, reference = self.model(), self.model()
        reference.requires_grad_(False)
        old = completion_logprobs(policy, [0, 1], [2]).detach().tolist()
        good = Trajectory([Turn([0, 1], [2], old)], 1.0, True, "terminated", {})
        empty = Trajectory([], 0.0, False, "context_limit", {})
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        stats = grpo_update(
            policy, reference, optimizer, [good, empty],
            beta=0.04, clip_epsilon=0.2, max_grad_norm=10,
        )
        advantage = group_advantages([1.0, 0.0])[0]
        # One active trajectory gets 1/2 group weight, including the empty failure.
        self.assertAlmostEqual(float(policy.table[1, 2]), 0.1 * advantage * 0.75 / 2, places=6)
        self.assertEqual(stats["empty_trajectories"], 1)
        self.assertEqual(stats["truncated"], 1)
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))

    def test_all_equal_rewards_have_zero_first_update(self):
        import torch
        policy, reference = self.model(), self.model()
        reference.requires_grad_(False)
        old = completion_logprobs(policy, [0], [1]).detach().tolist()
        trajectories = [
            Trajectory([Turn([0], [1], old)], 0.0, False, "terminated", {})
            for _ in range(2)
        ]
        before = policy.table.detach().clone()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        stats = grpo_update(
            policy, reference, optimizer, trajectories, beta=0.04, clip_epsilon=0.2
        )
        torch.testing.assert_close(policy.table, before)
        self.assertEqual(stats["zero_advantage_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
