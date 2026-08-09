#!/usr/bin/env python3
"""Adversarial tests for release archive parsing."""

from __future__ import annotations

import io
import gzip
import stat
import struct
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import release_archive
from release_archive import ArchiveError, safe_extract_tar_gz, safe_extract_zip


def write_canonical_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as payload, destination.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0
        ) as archive:
            archive.write(payload.read())


class ReleaseArchiveTests(unittest.TestCase):
    def test_zip_rejects_traversal_symlink_duplicates_and_extra_entries(self) -> None:
        cases = ("traversal", "symlink", "duplicate", "extra")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive_path = root / "hostile.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    if case == "traversal":
                        archive.writestr("../payload", b"x")
                    elif case == "symlink":
                        info = zipfile.ZipInfo("payload")
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        archive.writestr(info, b"target")
                    elif case == "duplicate":
                        archive.writestr("payload", b"one")
                        archive.writestr("payload", b"two")
                    else:
                        archive.writestr("payload", b"one")
                        archive.writestr("extra", b"two")
                with self.assertRaises(ArchiveError):
                    safe_extract_zip(archive_path, root / "out", ["payload"])

    def test_zip_rejects_excessive_compression_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "bomb.zip"
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("payload", bytes(2 * 1024 * 1024))
            with self.assertRaisesRegex(ArchiveError, "compression ratio"):
                safe_extract_zip(archive_path, root / "out", ["payload"])

    def test_zip_eocd_signature_inside_payload_or_comment_is_not_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "payload.zip"
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                archive.writestr("payload", b"ordinary PK\x05\x06 bytes")
            extracted = safe_extract_zip(archive_path, root / "out", ["payload"])
            self.assertEqual(extracted[0].read_bytes(), b"ordinary PK\x05\x06 bytes")

            comment_path = root / "comment.zip"
            with zipfile.ZipFile(comment_path, "w") as archive:
                archive.writestr("payload", b"ok")
                archive.comment = b"ordinary comment PK\x05\x06 bytes"
            # CPython's zipfile currently uses rfind() and rejects this legal
            # comment, but the bounded preflight itself must not call the
            # comment bytes a second EOCD record.
            release_archive._validate_zip_structure(comment_path)

    def test_zip_rejects_exotic_compression_and_encryption_flag(self) -> None:
        cases = ("bzip2", "encrypted")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive_path = root / "hostile.zip"
                compression = (
                    zipfile.ZIP_BZIP2 if case == "bzip2" else zipfile.ZIP_STORED
                )
                with zipfile.ZipFile(
                    archive_path, "w", compression=compression
                ) as archive:
                    archive.writestr("payload", b"ok")
                if case == "encrypted":
                    data = bytearray(archive_path.read_bytes())
                    central = data.find(b"PK\x01\x02")
                    self.assertGreaterEqual(central, 0)
                    flags = struct.unpack_from("<H", data, central + 8)[0]
                    struct.pack_into("<H", data, central + 8, flags | 0x1)
                    archive_path.write_bytes(data)
                with self.assertRaisesRegex(
                    ArchiveError, "compression method|encrypted"
                ):
                    safe_extract_zip(archive_path, root / "out", ["payload"])

    def test_zip_eocd_rejects_many_entries_zip64_oversized_central_and_trailing_data(self) -> None:
        cases = ("many", "zip64", "oversized-central", "trailing")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive_path = root / "hostile.zip"
                count = release_archive.MAX_ENTRIES + 1 if case == "many" else 1
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for index in range(count):
                        archive.writestr(f"payload-{index}", b"x")
                data = bytearray(archive_path.read_bytes())
                eocd = data.rfind(b"PK\x05\x06")
                self.assertGreaterEqual(eocd, 0)
                if case == "zip64":
                    struct.pack_into("<HH", data, eocd + 8, 0xFFFF, 0xFFFF)
                elif case == "oversized-central":
                    struct.pack_into(
                        "<L", data, eocd + 12, release_archive.MAX_CENTRAL_DIRECTORY_BYTES + 1
                    )
                elif case == "trailing":
                    data.extend(b"ambiguous")
                archive_path.write_bytes(data)
                with self.assertRaises(ArchiveError):
                    safe_extract_zip(
                        archive_path,
                        root / "out",
                        [f"payload-{index}" for index in range(count)],
                    )

    def test_tar_rejects_traversal_symlink_and_extra_entries(self) -> None:
        cases = ("traversal", "symlink", "extra")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive_path = root / "hostile.tar.gz"
                raw_tar = root / "hostile.tar"
                with tarfile.open(raw_tar, "w") as archive:
                    name = "../payload" if case == "traversal" else "payload"
                    info = tarfile.TarInfo(name)
                    if case == "symlink":
                        info.type = tarfile.SYMTYPE
                        info.linkname = "target"
                        archive.addfile(info)
                    else:
                        info.size = 1
                        archive.addfile(info, io.BytesIO(b"x"))
                    if case == "extra":
                        extra = tarfile.TarInfo("extra")
                        extra.size = 1
                        archive.addfile(extra, io.BytesIO(b"x"))
                write_canonical_gzip(raw_tar, archive_path)
                with self.assertRaises(ArchiveError):
                    safe_extract_tar_gz(archive_path, root / "out", ["payload"])

    def test_tar_gzip_stream_is_bounded_before_tar_parsing_or_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "bomb.tar.gz"
            raw_payload = root / "bomb.tar"
            raw_payload.write_bytes(bytes(8 * 1024))
            write_canonical_gzip(raw_payload, archive_path)
            with (
                mock.patch.object(release_archive, "MAX_TOTAL_BYTES", 1024),
                mock.patch.object(release_archive, "MAX_TAR_HEADER_BYTES", 512),
                self.assertRaisesRegex(ArchiveError, "expanded stream exceeds"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])
            self.assertFalse((root / "out").exists())

    def test_tar_declared_member_size_is_checked_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "huge-member.tar.gz"
            raw_tar = root / "huge-member.tar"
            with tarfile.open(raw_tar, "w") as archive:
                info = tarfile.TarInfo("payload")
                info.size = 2
                archive.addfile(info, io.BytesIO(b"xx"))
            write_canonical_gzip(raw_tar, archive_path)
            with (
                mock.patch.object(release_archive, "MAX_ENTRY_BYTES", 1),
                self.assertRaisesRegex(ArchiveError, "size bound"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])
            self.assertFalse((root / "out").exists())

    def test_tar_preflight_rejects_thousands_of_headers_before_tarfile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "many.tar.gz"
            raw_tar = root / "many.tar"
            with tarfile.open(raw_tar, "w") as archive:
                for index in range(1000):
                    archive.addfile(tarfile.TarInfo(f"entry-{index}"))
            write_canonical_gzip(raw_tar, archive_path)
            with (
                mock.patch.object(release_archive, "MAX_COMPRESSION_RATIO", 10000),
                mock.patch.object(
                    release_archive.tarfile,
                    "open",
                    side_effect=AssertionError("tarfile must not parse hostile metadata"),
                ),
                self.assertRaisesRegex(ArchiveError, "too many entries"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])
            self.assertFalse((root / "out").exists())

    def test_tar_preflight_rejects_huge_pax_extension_before_tarfile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "pax.tar.gz"
            raw_tar = root / "pax.tar"
            with tarfile.open(
                raw_tar,
                "w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": "x" * (512 * 1024)},
            ) as archive:
                info = tarfile.TarInfo("payload")
                info.size = 2
                archive.addfile(info, io.BytesIO(b"ok"))
            write_canonical_gzip(raw_tar, archive_path)
            with (
                mock.patch.object(release_archive, "MAX_COMPRESSION_RATIO", 10000),
                mock.patch.object(
                    release_archive.tarfile,
                    "open",
                    side_effect=AssertionError("tarfile must not parse hostile metadata"),
                ),
                self.assertRaisesRegex(ArchiveError, "extensions"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])
            self.assertFalse((root / "out").exists())

    def test_tar_rejects_hidden_nonzero_data_after_last_member(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            raw_tar = root / "hostile.tar"
            with tarfile.open(raw_tar, "w") as archive:
                info = tarfile.TarInfo("payload")
                info.size = 2
                archive.addfile(info, io.BytesIO(b"ok"))
            with raw_tar.open("ab") as archive:
                archive.write(b"HIDDEN" + bytes(506))
            archive_path = root / "hostile.tar.gz"
            write_canonical_gzip(raw_tar, archive_path)
            with self.assertRaisesRegex(ArchiveError, "hidden data"):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])
            self.assertFalse((root / "out").exists())

    def test_gzip_preflight_rejects_huge_filename_before_zlib(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "fname.tar.gz"
            archive_path.write_bytes(
                b"\x1f\x8b\x08\x08" + bytes(6) + (b"x" * (1024 * 1024))
            )
            with (
                mock.patch.object(
                    release_archive.zlib,
                    "decompressobj",
                    side_effect=AssertionError("zlib must not parse gzip metadata"),
                ),
                self.assertRaisesRegex(ArchiveError, "metadata-free gzip header"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])

    def test_gzip_preflight_rejects_concatenated_members_before_tarfile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "concatenated.tar.gz"
            archive_path.write_bytes(
                gzip.compress(b"first", mtime=0) + gzip.compress(b"second", mtime=0)
            )
            with (
                mock.patch.object(
                    release_archive.tarfile,
                    "open",
                    side_effect=AssertionError("tarfile must not parse concatenated gzip"),
                ),
                self.assertRaisesRegex(ArchiveError, "concatenated members"),
            ):
                safe_extract_tar_gz(archive_path, root / "out", ["payload"])

    def test_safe_zip_extracts_only_expected_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "safe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("payload", b"ok")
            extracted = safe_extract_zip(archive_path, root / "out", ["payload"])
            self.assertEqual([path.read_bytes() for path in extracted], [b"ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
