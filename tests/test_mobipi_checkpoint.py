import unittest

import numpy as np

from mobiwam.mobipi_checkpoint import normalization_stats_to_numpy


class MobiPiCheckpointTest(unittest.TestCase):
    def test_checkpoint_normalization_lists_are_restored_as_arrays(self):
        stats = normalization_stats_to_numpy(
            {"actions": {"scale": [[1.0] * 12], "offset": [[0.0] * 12]}}
        )
        self.assertIsNotNone(stats)
        self.assertEqual(stats["actions"]["scale"].shape, (1, 12))
        self.assertEqual(stats["actions"]["offset"].shape, (1, 12))
        np.testing.assert_allclose(stats["actions"]["scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
