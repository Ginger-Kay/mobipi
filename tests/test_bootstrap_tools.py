import importlib.util
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BootstrapToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = load_script_module(
            "download_mobipi_close_single_door",
            "scripts/download_mobipi_close_single_door.py",
        )
        cls.configure = load_script_module(
            "configure_mobipi_paths",
            "scripts/configure_mobipi_paths.py",
        )
        cls.preflight = load_script_module(
            "mobipi_preflight",
            "scripts/mobipi_preflight.py",
        )

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            with zipfile.ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(RuntimeError, "unsafe ZIP path"):
                    self.downloader.validate_zip_members(archive)

    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "symlink.zip"
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(info, "target")
            with zipfile.ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(RuntimeError, "symbolic links"):
                    self.downloader.validate_zip_members(archive)

    def test_verified_artifact_extracts_to_expected_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "artifact.zip"
            payload = b"verified payload"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("top/sub/file.bin", payload)

            artifact = self.downloader.Artifact(
                name="fixture",
                filename="artifact.zip",
                sha256="unused-by-extractor",
                extract_root="data",
                expected_relative_path="top/sub/file.bin",
                expected_size=len(payload),
            )
            result = self.downloader.extract_artifact(artifact, archive_path, root)
            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(any((root / "downloads" / "mobipi" / "staging").iterdir()))

    def test_atomic_macro_create_refuses_different_existing_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "macros_private.py"
            self.assertEqual(self.configure.atomic_create(target, "VALUE = 1\n"), "created")
            self.assertEqual(self.configure.atomic_create(target, "VALUE = 1\n"), "unchanged")
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                self.configure.atomic_create(target, "VALUE = 2\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 1\n")

    def test_ready_does_not_require_source_hdf5_but_full_does(self):
        self.assertFalse(self.preflight.stage_requires_dataset("environment"))
        self.assertFalse(self.preflight.stage_requires_dataset("ready"))
        self.assertTrue(self.preflight.stage_requires_dataset("full"))
        with self.assertRaisesRegex(ValueError, "unknown preflight stage"):
            self.preflight.stage_requires_dataset("invalid")

    def test_robocasa365_port_config_is_pinned_and_isolated(self):
        config = json.loads(
            (PROJECT_ROOT / "configs" / "robocasa365_port.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["release_name"], "RoboCasa365 v1.0.1")
        self.assertEqual(len(config["robocasa_commit"]), 40)
        self.assertEqual(len(config["robosuite_commit"]), 40)
        self.assertNotEqual(config["robocasa_repo"], "/share/chensiyu/MobiWAM/repos/mobipi")


if __name__ == "__main__":
    unittest.main()
