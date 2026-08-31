import json
import tempfile
import unittest
from pathlib import Path

from mobiwam.run_config import bind_run_config


class RunConfigBindingTest(unittest.TestCase):
    def test_exact_commit_is_bound_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.json"
            output = root / "run" / "config.json"
            template.write_text(
                json.dumps({"code_commit": "BIND_AT_RUN", "x": 1}),
                encoding="utf-8",
            )
            checksum = bind_run_config(template, output, code_commit="a" * 40)
            self.assertEqual(json.loads(output.read_text())["code_commit"], "a" * 40)
            self.assertEqual(len(checksum), 64)
            with self.assertRaises(FileExistsError):
                bind_run_config(template, output, code_commit="b" * 40)


if __name__ == "__main__":
    unittest.main()
