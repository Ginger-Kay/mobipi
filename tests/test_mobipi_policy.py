import unittest

import numpy as np

from mobiwam.mobipi_policy import sample_verified_future_chunk


class FakeNetwork:
    def __init__(self, chunk):
        self.chunk = np.asarray(chunk, dtype=np.float64)
        self.call_count = 0

    def __call__(self, *, obs_dict, actions, goal_dict):
        del obs_dict, actions, goal_dict
        self.call_count += 1
        return self.chunk[np.newaxis] + 0.026 * (self.call_count - 1)


class FakeAlgo:
    supervise_all_steps = True
    pred_future_acs = True
    context_length = 10

    def __init__(self, chunk):
        self.nets = {"policy": FakeNetwork(chunk)}

    def get_action(self, obs_dict, goal_dict=None):
        return self.nets["policy"](
            obs_dict=obs_dict, actions=None, goal_dict=goal_dict
        )[:, 0, :]


class FakeRolloutPolicy:
    def __init__(self, chunk):
        self.policy = FakeAlgo(chunk)

    def __call__(self, observation, goal=None, batched=False):
        del batched
        value = self.policy.get_action(obs_dict=observation, goal_dict=goal)[0]
        return value * 2.0 + 3.0


class FutureChunkTest(unittest.TestCase):
    def test_exposure_uses_one_forward_and_matches_official_first_action(self):
        raw = np.arange(120, dtype=np.float64).reshape(10, 12) / 100.0
        policy = FakeRolloutPolicy(raw)
        evidence = sample_verified_future_chunk(policy, {"x": 1})
        np.testing.assert_allclose(evidence.chunk, raw * 2.0 + 3.0)
        np.testing.assert_allclose(evidence.chunk[0], evidence.official_first_action)
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.max_abs_error, 0.0)
        self.assertEqual(policy.policy.nets["policy"].call_count, 1)


if __name__ == "__main__":
    unittest.main()
