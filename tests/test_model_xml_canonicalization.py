import unittest

import numpy as np

from mobiwam.adapters.mobipi import _state_hash


class ModelXmlCanonicalizationTest(unittest.TestCase):
    def _state(self, model: str) -> dict:
        return {
            "model": model,
            "states": np.array([1.0, 2.0]),
            "ep_meta": '{"lang":"close the door","layout_id":1}',
        }

    def test_explicit_identity_refquat_matches_default(self):
        explicit = self._state(
            '<mesh name="cup" refquat="1 0 0 0" scale="0.1 0.1 0.1"/>'
        )
        implicit = self._state('<mesh name="cup" scale="0.1 0.1 0.1"/>')
        self.assertEqual(_state_hash(explicit), _state_hash(implicit))

    def test_non_identity_refquat_remains_significant(self):
        identity = self._state('<mesh name="cup" refquat="1 0 0 0"/>')
        rotated = self._state('<mesh name="cup" refquat="0.707 0 0 0.707"/>')
        self.assertNotEqual(_state_hash(identity), _state_hash(rotated))


if __name__ == "__main__":
    unittest.main()
