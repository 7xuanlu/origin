#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).with_name("post-write-move-check.py")
SPEC = importlib.util.spec_from_file_location("post_write_move_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PostWriteMoveCheckTests(unittest.TestCase):
    def test_declared_normalizations_are_narrow(self):
        MODULE.selftest()

    def test_non_allowlisted_token_mutation_changes_digest(self):
        original = "async fn stable() {\n    write(expected);\n}"
        mutated = "async fn stable() {\n    write(unexpected);\n}"
        self.assertNotEqual(
            MODULE.normalize_item(
                original, dedent=False, allow_pub_super=False
            ),
            MODULE.normalize_item(
                mutated, dedent=False, allow_pub_super=False
            ),
        )

    def test_raw_string_mask_stops_at_its_exact_delimiter(self):
        source = 'const RAW: &str = r##"{ not code }"##;\nfn visible() {}\n'
        masked = MODULE.mask_rust(source)
        self.assertNotIn("not code", masked)
        self.assertIn("fn visible() {}", masked)

    def test_unicode_before_item_and_import_preserves_exact_offsets(self):
        source = (
            "//! naïve → 測試\n"
            'const LABEL: &str = "résumé";\n'
            "use crate::{Alpha, Beta};\n"
            "async fn visible() {\n"
            "    work();\n"
            "}\n"
            "fn after() {}\n"
        )
        selected = MODULE.extract_selector(
            source, {"kind": "fn", "name": "visible"}
        )
        self.assertEqual(
            selected,
            "async fn visible() {\n    work();\n}",
        )
        self.assertEqual(
            MODULE.top_level_imports(source),
            {"use crate::{Alpha, Beta};"},
        )


if __name__ == "__main__":
    unittest.main()
