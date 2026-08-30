import unittest

import numpy as np

from mobiwam.dock_protocol import settle_flush_and_reset_policy


class FakeEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action))
        return {}, 0.0, False, {}


class FakePolicy:
    def __init__(self):
        self.languages = []

    def start_episode(self, lang=None):
        self.languages.append(lang)


class DockProtocolTest(unittest.TestCase):
    def test_requires_ten_consecutive_stable_zero_action_frames(self):
        env = FakeEnv()
        policy = FakePolicy()
        speeds = iter([(0.1, 0.1), (0.0, 0.0)] * 2 + [(0.0, 0.0)] * 10)
        evidence = settle_flush_and_reset_policy(
            env,
            policy,
            language="close the door",
            velocity_reader=lambda _: next(speeds),
            history_length=10,
            max_steps=20,
        )
        self.assertEqual(evidence.steps, 13)
        self.assertEqual(evidence.stable_zero_action_frames, 10)
        self.assertEqual(policy.languages, ["close the door"])
        self.assertEqual(len(env.actions), 13)
        for action in env.actions:
            np.testing.assert_allclose(action, 0.0)

    def test_timeout_does_not_reset_policy(self):
        env = FakeEnv()
        policy = FakePolicy()
        with self.assertRaises(RuntimeError):
            settle_flush_and_reset_policy(
                env,
                policy,
                language="close the door",
                velocity_reader=lambda _: (0.1, 0.1),
                history_length=3,
                max_steps=3,
            )
        self.assertEqual(policy.languages, [])


if __name__ == "__main__":
    unittest.main()
