#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ci-timed-command.py"


class TimedCommandTests(unittest.TestCase):
    def test_success_writes_timing_and_output_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "artifact.bin"
            receipt = root / "receipt.json"
            target = root / "target"
            target.mkdir()
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(target)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--label",
                    "probe",
                    "--output",
                    str(receipt),
                    "--hash",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(output)!r}).write_bytes(b'abc')",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["exit_code"], 0)
            self.assertGreaterEqual(payload["duration_seconds"], 0)
            self.assertEqual(
                payload["outputs"][0]["sha256"],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(payload["disk"]["path"], str(target))

    def test_failure_is_returned_after_receipt_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--label",
                    "failure",
                    "--output",
                    str(receipt),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7)
            self.assertEqual(json.loads(receipt.read_text())["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
