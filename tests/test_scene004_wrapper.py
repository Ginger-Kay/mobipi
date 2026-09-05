from collections import deque

from scripts.b0_scene004_u2_scan import initialize_constructor_frame_stack


class FakeInner:
    def __init__(self, events):
        self.events = events

    def get_observation(self):
        self.events.append("get_observation")
        return {"x": 1}

    def get_ep_meta(self):
        self.events.append("get_ep_meta")
        return {"lang": "close fixture"}


class FakeWrapper:
    def __init__(self):
        self.events = []
        self.env = FakeInner(self.events)
        self.unwrapped = self
        self.reset_calls = 0
        self.step_calls = 0

    def update_obs(self, observation, reset):
        assert self.timestep == 0
        assert self._ep_lang_str == "close fixture"
        assert reset is True
        self.events.append("update_obs")

    def _get_initial_obs_history(self, observation):
        self.events.append("initial_history")
        return {"x": deque([observation["x"]] * 10, maxlen=10)}

    def _get_stacked_obs_from_history(self):
        self.events.append("stacked_history")
        return {"x": list(self.obs_history["x"])}


def test_constructor_wrapper_metadata_precedes_update_and_does_not_reset_or_step():
    wrapper = FakeWrapper()
    stacked = initialize_constructor_frame_stack(wrapper)
    assert wrapper.events == ["get_ep_meta", "get_observation", "update_obs", "initial_history", "stacked_history"]
    assert len(stacked["x"]) == 10
    assert wrapper.reset_calls == wrapper.step_calls == 0
