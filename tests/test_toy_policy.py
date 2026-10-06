"""Meaningful numerical and data-boundary tests for the CPU toy policy."""
import copy
import json
import unittest
import numpy as np
from bank_agent.env import BankEnv
from bank_agent.policies import decide
from bank_agent.toy_policy import (
    TinyPolicy, Adam, ACTION_NAMES, FEATURE_NAMES, features, bind_action,
    group_advantages, fit_supervised, expert_dataset, fit_group_reinforce,
)


def task():
    return {
        "task_id": "train-0", "split": "train", "case_id": "case-0",
        "request_id": "request-0", "request": "Please handle my case",
        "role": "employee", "principal_id": "staff-0", "owner_id": "customer-0",
        "assigned_employee_id": "staff-0", "family": "billing",
        "has_materials": False, "requires_materials": True, "amount": 150,
        "threshold": 100, "policy_version": "v1", "faults": [],
    }


class ToyPolicyTests(unittest.TestCase):
    def test_hidden_faults_and_target_do_not_enter_features(self):
        first, second = task(), task()
        second.update(faults=["read_timeout", "policy_change"], amount=9999,
                      new_threshold=1, new_policy_version="v2")
        a, b = BankEnv(first), BankEnv(second)
        np.testing.assert_array_equal(features(a.messages), features(b.messages))
        self.assertEqual(len(features(a.messages)), len(FEATURE_NAMES))

    def test_argument_binding_only_uses_observed_policy_and_case(self):
        env = BankEnv(task())
        before = bind_action(ACTION_NAMES.index("resolve_case"), env.messages)
        self.assertEqual(before["arguments"]["policy_version"], "unknown")
        env.step(decide(env.messages))
        env.step(decide(env.messages))
        after = bind_action(ACTION_NAMES.index("resolve_case"), env.messages)
        self.assertEqual(after["arguments"]["policy_version"], "v1")
        self.assertEqual(after["arguments"]["idempotency_key"], "request-0")
        self.assertEqual(after["arguments"]["resolution"], "escalated")

    def test_cross_entropy_gradient_matches_finite_difference(self):
        model = TinyPolicy(seed=4, hidden=5)
        rng = np.random.default_rng(9)
        x = rng.normal(size=(4, len(FEATURE_NAMES)))
        labels = np.array([0, 2, 4, 7])
        _, grads = model.supervised_loss_grad(x, labels)
        for name, index in [("w1", (2, 3)), ("b1", (1,)), ("w2", (3, 6)), ("b2", (0,))]:
            old = model.params[name][index]
            model.params[name][index] = old + 1e-6
            plus = model.supervised_loss_grad(x, labels)[0]
            model.params[name][index] = old - 1e-6
            minus = model.supervised_loss_grad(x, labels)[0]
            model.params[name][index] = old
            self.assertAlmostEqual(grads[name][index], (plus - minus) / 2e-6, places=6)

    def test_reinforce_and_exact_kl_gradient_matches_finite_difference(self):
        model = TinyPolicy(seed=4, hidden=5)
        reference = TinyPolicy(seed=7, hidden=5)
        rng = np.random.default_rng(9)
        x = rng.normal(size=(4, len(FEATURE_NAMES)))
        chosen = [1, 2, 3, 4]
        advantages = [-1, 1, 0.5, -0.5]
        _, grads, _ = model.reinforce_loss_grad(x, chosen, advantages, reference, 0.1, 2)
        for name, index in [("w1", (2, 3)), ("b1", (1,)), ("w2", (3, 6)), ("b2", (0,))]:
            old = model.params[name][index]
            model.params[name][index] = old + 1e-6
            plus = model.reinforce_loss_grad(x, chosen, advantages, reference, 0.1, 2)[0]
            model.params[name][index] = old - 1e-6
            minus = model.reinforce_loss_grad(x, chosen, advantages, reference, 0.1, 2)[0]
            model.params[name][index] = old
            self.assertAlmostEqual(grads[name][index], (plus - minus) / 2e-6, places=6)

    def test_positive_advantage_increases_action_probability(self):
        model = TinyPolicy(seed=4)
        x = features(BankEnv(task()).messages)[None, :]
        selected = [0]
        before = model.probabilities(x)[0, 0]
        _, grads, _ = model.reinforce_loss_grad(x, selected, [1.0], model.clone(), 0, 1)
        Adam(model.params, 0.001).update(model.params, grads)
        self.assertGreater(model.probabilities(x)[0, 0], before)

    def test_equal_rewards_have_no_policy_gradient_signal(self):
        np.testing.assert_array_equal(group_advantages([1, 1, 1]), np.zeros(3))
        model = TinyPolicy(seed=5)
        x = features(BankEnv(task()).messages)[None, :]
        _, grads, _ = model.reinforce_loss_grad(x, [0], [0], model.clone(), 0.02, 1)
        self.assertLess(max(np.abs(g).max() for g in grads.values()), 1e-12)

    def test_sft_reduces_loss_on_actual_expert_states(self):
        x, y, _ = expert_dataset([task()])
        model = TinyPolicy(seed=8)
        log = fit_supervised(model, x, y, seed=8, steps=80, batch_size=32)
        self.assertLess(log[-1]["full_training_loss"], log[0]["full_training_loss"] / 5)

    def test_train_apis_reject_held_out_data(self):
        held_out = task()
        held_out["split"] = "test"
        with self.assertRaises(ValueError):
            expert_dataset([held_out])
        with self.assertRaises(ValueError):
            fit_group_reinforce(TinyPolicy(), [held_out], updates=1)


if __name__ == "__main__":
    unittest.main()
