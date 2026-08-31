import tempfile
import unittest
from pathlib import Path

import numpy as np

from mobiwam.feature_cache import FeatureCache, feature_cache_key


class SharedFeatureCacheTest(unittest.TestCase):
    def test_atomic_round_trip_and_checksum_key(self):
        key = feature_cache_key(
            source_checksum="source-sha",
            candidate_checksum="candidate-sha",
            encoder_revision="encoder-revision",
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = FeatureCache(Path(temporary))
            array = np.arange(12, dtype=np.float32).reshape(3, 4)
            record = cache.put(key, array)
            self.assertEqual(record.key, key)
            np.testing.assert_array_equal(cache.get(key), array)
            self.assertEqual(list(Path(temporary).glob("*.partial-*")), [])


if __name__ == "__main__":
    unittest.main()
