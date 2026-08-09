import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify-ort-source-pin.py")
SPEC = importlib.util.spec_from_file_location("verify_ort_source_pin", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyOrtSourcePinTests(unittest.TestCase):
    def make_source(
        self,
        *,
        commit=MODULE.EXPECTED_COMMIT,
        dist_version=MODULE.EXPECTED_ORT_VERSION,
        api_version=MODULE.EXPECTED_API_VERSION,
    ):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "build/download").mkdir(parents=True)
        (root / "src").mkdir()
        (root / ".cargo_vcs_info.json").write_text(
            json.dumps(
                {
                    "git": {"sha1": commit, "dirty": True},
                    "path_in_vcs": "ort-sys",
                }
            ),
            encoding="utf-8",
        )
        (root / "build/download/dist.txt").write_text(
            "none\tx86_64-pc-windows-msvc\t"
            f"https://cdn.pyke.io/0/pyke:ort-rs/ms@{dist_version}/"
            "x86_64-pc-windows-msvc.tar.lzma2\tdeadbeef\n",
            encoding="utf-8",
        )
        (root / "src/version.rs").write_text(
            f"pub const ORT_API_VERSION: u32 = {api_version};\n",
            encoding="utf-8",
        )
        return temp, root

    def test_accepts_exact_published_source_contract(self):
        temp, root = self.make_source()
        self.addCleanup(temp.cleanup)
        self.assertEqual(MODULE.verify_ort_sys_source(root), [])

    def test_rejects_wrong_commit_runtime_and_api_version(self):
        temp, root = self.make_source(
            commit="0" * 40,
            dist_version="1.22.0",
            api_version=22,
        )
        self.addCleanup(temp.cleanup)
        violations = MODULE.verify_ort_sys_source(root)
        self.assertTrue(any("commit" in violation for violation in violations))
        self.assertTrue(any("1.23.2" in violation for violation in violations))
        self.assertTrue(any("API version 23" in violation for violation in violations))


if __name__ == "__main__":
    unittest.main()
