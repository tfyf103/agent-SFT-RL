"""CPU contract tests, NOT evidence of an LLM/GPU training run.

The fake sampler returns indivisible synthetic output tokens; attempting to
re-tokenize generated text fails. All fake Slime modules use monkeypatch and are
restored after each test, including when a real Slime install is present.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
import types
from enum import Enum

import pytest

from bank_agent.data import generate_tasks
from bank_agent.policies import decide
from integrations.slime_bank import generate, parse_action


@pytest.fixture
def fake_slime(monkeypatch):
    class Tokenizer:
        def __init__(self):
            self.generated = {}

        def encode(self, text, add_special_tokens=False):
            assert text not in self.generated.values(), "Generated text was re-tokenized"
            tokens = []
            for part in re.split(r"(<\|im_(?:start|end)\|>)", text):
                if part == "<|im_start|>":
                    tokens.append(1)
                elif part == "<|im_end|>":
                    tokens.append(2)
                else:
                    tokens.extend(ord(char) + 200 for char in part)
            return tokens

        def decode(self, tokens, skip_special_tokens=False):
            return "".join(
                "<|im_start|>" if token == 1 else
                "<|im_end|>" if token == 2 else
                self.generated[token] if token in self.generated else
                chr(token - 200)
                for token in tokens
            )

        def apply_chat_template(self, messages, **kwargs):
            return "".join(
                "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
                for message in messages
            ) + "<|im_start|>assistant\n"

    tokenizer = Tokenizer()
    state = types.SimpleNamespace(finish_reason="stop", calls=0, tokenizer=tokenizer)

    class Sample:
        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"

        def __init__(self, task):
            self.index = 41
            self.group_index = 3
            self.rollout_id = 9
            self.metadata = {"task": copy.deepcopy(task), "preserve": "yes"}
            self.response_length = 0

        def append_response_tokens(
            self, args, tokens, log_probs=None, trainable=True,
            meta_info=None, text=None, update_terminal_info=False,
        ):
            assert len(log_probs or []) == len(tokens) if trainable else log_probs is None
            self.tokens += tokens
            self.response_length += len(tokens)
            self.loss_mask += [int(trainable)] * len(tokens)
            self.rollout_log_probs += log_probs if trainable else [0.0] * len(tokens)
            self.response += text or ""

    class GenerateState:
        def __init__(self, args):
            self.tokenizer = tokenizer

    async def post(url, payload):
        state.calls += 1
        assert "input_ids" in payload and "text" not in payload
        assert payload["return_logprob"] is True
        assert payload["sampling_params"]["no_stop_trim"] is True
        if state.finish_reason == "abort":
            return {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}
        transcript = tokenizer.decode(payload["input_ids"])
        messages = [
            {"role": role, "content": content}
            for role, content in re.findall(
                r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
                transcript, re.S,
            )
        ]
        text = json.dumps(decide(messages), ensure_ascii=False)
        token_id = 500000 + len(tokenizer.generated)
        tokenizer.generated[token_id] = text
        return {
            "text": text + "<|im_end|>",
            "meta_info": {
                "finish_reason": {"type": state.finish_reason},
                "output_token_logprobs": [[-0.1, token_id], [-0.2, 2]],
            },
        }

    for name in ("slime", "slime.rollout", "slime.utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name, attributes in (
        ("slime.rollout.sglang_rollout", {"GenerateState": GenerateState}),
        ("slime.utils.http_utils", {"post": post}),
        ("slime.utils.types", {"Sample": Sample}),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    state.Sample = Sample
    state.args = types.SimpleNamespace(
        partial_rollout=False, use_score_centering=False,
        sglang_router_ip="127.0.0.1", sglang_router_port=30000,
    )
    return state


@pytest.mark.parametrize("split", ["train", "valid", "test", "stress"])
def test_full_trajectories_preserve_tokens_masks_and_identity(fake_slime, split):
    """80 complete trajectories, covering both roles and fault combinations."""
    state = fake_slime

    async def run():
        for task in generate_tasks(split, 20):
            original = state.Sample(task)
            sample = await generate(
                state.args, original, {"max_new_tokens": 20000},
                evaluation=split != "train",
            )
            assert sample is original
            assert (sample.index, sample.group_index, sample.rollout_id) == (41, 3, 9)
            assert sample.metadata["preserve"] == "yes"
            assert sample.metadata["task"] == task
            assert sample.reward == 1.0, sample.metadata["bank_result"]
            assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
            assert set(sample.loss_mask) == {0, 1}
            response_tokens = sample.tokens[-sample.response_length:]
            for token, mask, logprob in zip(
                response_tokens, sample.loss_mask, sample.rollout_log_probs
            ):
                if token in state.tokenizer.generated:
                    assert mask == 1 and logprob == -0.1
                if not mask:
                    assert logprob == 0.0

    asyncio.run(run())


def test_truncated_sampler_group_receives_only_zero_rewards(fake_slime):
    """Even valid-looking JSON must not execute when the sampler says length."""
    state = fake_slime
    state.finish_reason = "length"

    async def run():
        rewards = []
        for task in generate_tasks("train", 4):
            sample = await generate(state.args, state.Sample(task), {"max_new_tokens": 20000})
            assert sample.status == state.Sample.Status.TRUNCATED
            assert sample.metadata["bank_trace"] == []
            assert sample.metadata["bank_result"]["steps"] == 0
            assert sample.response_length == 2
            assert sample.loss_mask == [1, 1]
            assert sample.rollout_log_probs == [-0.1, -0.2]
            rewards.append(sample.reward)
        assert rewards == [0.0] * 4

    asyncio.run(run())


def test_response_budget_truncation_has_no_false_success(fake_slime):
    state = fake_slime
    sample = asyncio.run(generate(
        state.args, state.Sample(generate_tasks("train", 1)[0]),
        {"max_new_tokens": 2},
    ))
    assert sample.status == state.Sample.Status.TRUNCATED
    assert sample.reward == 0.0
    assert sample.metadata["bank_result"]["writes"] == 0
    assert sample.response_length == 2
    assert len(sample.loss_mask) == len(sample.rollout_log_probs) == 2


@pytest.mark.parametrize("split", ["valid", "test", "stress"])
def test_heldout_tasks_rejected_before_sampling(fake_slime, split):
    state = fake_slime
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(generate(
            state.args, state.Sample(generate_tasks(split, 1)[0]),
            {"max_new_tokens": 20000}, evaluation=False,
        ))
    assert state.calls == 0


def test_aborted_sampler_never_creates_a_reward(fake_slime):
    state = fake_slime
    state.finish_reason = "abort"
    sample = asyncio.run(generate(
        state.args, state.Sample(generate_tasks("train", 1)[0]),
        {"max_new_tokens": 20000},
    ))
    assert sample.status == state.Sample.Status.ABORTED
    assert sample.reward is None
    assert sample.response_length == 0
    assert "bank_result" not in sample.metadata


def test_invalid_action_parser():
    assert parse_action("not JSON")["tool"] == "__invalid__"
    assert parse_action("[]")["tool"] == "__invalid__"
    assert parse_action('{"tool":"get_status","arguments":{}}<|im_end|>')["tool"] == "get_status"
