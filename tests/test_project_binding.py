import unittest
from pathlib import Path


PROJECT_ROOT = Path("/share/jhk/MobiWAM")
REPO_ROOT = PROJECT_ROOT / "Mobipi"


class ProjectBindingTest(unittest.TestCase):
    def test_canonical_flat_roots(self):
        expected = {
            "control",
            "Mobipi",
            "env",
            "data",
            "checkpoints",
            "artifacts",
            "cache",
            "imports",
        }
        self.assertTrue(expected.issubset({path.name for path in PROJECT_ROOT.iterdir()}))
        self.assertFalse((PROJECT_ROOT / "code" / "mobipi").exists())
        self.assertFalse((PROJECT_ROOT / "mobipi").exists())
        self.assertFalse((PROJECT_ROOT / "envs" / "mobiwam").exists())

    def test_runtime_sources_do_not_reference_donor_or_preemption(self):
        forbidden = (
            "/share/chensiyu/MobiWAM",
            "gpu_with_lease",
            "MOBIWAM_GPU_LEASE",
            "killall",
            "pkill",
            "SIGTERM",
            "SIGKILL",
        )
        checked = []
        for relative in ("configs", "scripts", "src", "tests"):
            for path in sorted((REPO_ROOT / relative).rglob("*")):
                if not path.is_file() or path.suffix in {".pyc", ".png"}:
                    continue
                if path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                checked.append(path)
                for token in forbidden:
                    self.assertNotIn(token, text, f"{token!r} leaked into {path}")
        self.assertTrue(checked)


if __name__ == "__main__":
    unittest.main()
