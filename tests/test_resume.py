"""A resumed optimizer run must match uninterrupted training, not just load."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ResumeTests(unittest.TestCase):
    def test_resumed_training_has_the_same_weights_and_losses(self):
        base = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def train(name, steps, resume=False):
                output = root / name
                command = [sys.executable, "-m", "ellmiott_gpt.train", "--steps", str(steps),
                           "--early-step", "1", "--eval-every", "1", "--eval-batches", "1",
                           "--batch-size", "2", "--threads", "1", "--output", str(output / "models"),
                           "--run-dir", str(output / "run"), "--fixtures", str(output / "parity.json")]
                if resume:
                    command += ["--resume", str(output / "run/training.pt")]
                result = subprocess.run(command, cwd=base, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return output

            full = train("full", 4)
            train("resumed", 2)
            resumed = train("resumed", 4, resume=True)
            self.assertEqual((full / "models/trained.json").read_bytes(), (resumed / "models/trained.json").read_bytes())
            a = json.loads((full / "models/manifest.json").read_text())
            b = json.loads((resumed / "models/manifest.json").read_text())
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["checkpoints"], b["checkpoints"])
            self.assertEqual(a["training"]["steps"], 4)
            self.assertEqual(b["training"]["steps"], 4)
            self.assertEqual((full / "parity.json").read_bytes(), (resumed / "parity.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
