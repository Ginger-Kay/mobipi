from pathlib import Path

from mobiwam.scene004_renderer import NATIVE_HEIGHT, NATIVE_WIDTH


def test_native_renderer_dimensions_are_frozen():
    assert (NATIVE_WIDTH, NATIVE_HEIGHT) == (1920, 1080)


def test_renderer_worker_imports_only_after_runtime_environment_setup():
    text = Path(__file__).parents[1].joinpath("scripts/b0_scene004_renderer_worker.py").read_text()
    assert text.index('os.environ["MUJOCO_GL"]') < text.index("from mobiwam.scene004_renderer import")


def test_renderer_module_has_zero_task_environment_calls():
    text = Path(__file__).parents[1].joinpath("src/mobiwam/scene004_renderer.py").read_text()
    assert "env.reset" not in text
    assert "env.step" not in text
    assert "_check_success" not in text
