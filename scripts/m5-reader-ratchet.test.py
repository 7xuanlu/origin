#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).with_name("m5-reader-ratchet.py")
SPEC = importlib.util.spec_from_file_location("m5_reader_ratchet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class M5ReaderRatchetTests(unittest.TestCase):
    def test_synthetic_summary_and_exposure_mutations_fail(self):
        MODULE.selftest()

    def test_pinned_exposure_count_is_exact(self):
        self.assertEqual(len(MODULE.PINNED_EXPOSURES), 22)


if __name__ == "__main__":
    unittest.main()
