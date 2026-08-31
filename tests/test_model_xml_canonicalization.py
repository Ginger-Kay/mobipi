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

    def test_compiled_geom_quaternion_rounding_is_canonicalized(self):
        captured = self._state(
            '<geom name="fixture" quat="0.955336 -0.29552 0 0" type="box"/>'
        )
        restored = self._state(
            '<geom name="fixture" quat="0.955337 -0.29552 0 0" type="box"/>'
        )
        self.assertEqual(_state_hash(captured), _state_hash(restored))

    def test_material_geom_quaternion_change_remains_significant(self):
        first = self._state(
            '<geom name="fixture" quat="0.955336 -0.29552 0 0" type="box"/>'
        )
        second = self._state(
            '<geom name="fixture" quat="0.95 -0.31 0 0" type="box"/>'
        )
        self.assertNotEqual(_state_hash(first), _state_hash(second))


if __name__ == "__main__":
    unittest.main()
