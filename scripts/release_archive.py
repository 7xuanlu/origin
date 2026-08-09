#!/usr/bin/env python3
"""Bounded archive creation and extraction primitives for release payloads."""

from __future__ import annotations

import gzip
import os
import shutil
import stat
import struct
import tarfile
import tempfile
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath


MAX_ARCHIVE_BYTES = 384 * 1024 * 1024
MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024
MAX_ENTRIES = 32
MAX_COMPRESSION_RATIO = 200
MAX_CENTRAL_DIRECTORY_BYTES = 256 * 1024
MAX_TAR_HEADER_BYTES = 2 * 1024 * 1024


class ArchiveError(ValueError):
    """An archive is not the exact bounded regular-file payload expected."""


def _require_plain_name(name: str) -> str:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or PurePosixPath(name).is_absolute()
        or PurePosixPath(name).name != name
        or name in {".", ".."}
    ):
        raise ArchiveError(f"unsafe archive entry name {name!r}")
    return name


def _require_expected(names: list[str], expected: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ArchiveError("archive contains duplicate entry names")
    if set(names) != set(expected) or len(names) != len(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ArchiveError(f"archive inventory mismatch: missing={missing} extra={extra}")


def _copy_bounded(source, destination, expected_size: int) -> None:
    copied = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        copied += len(chunk)
        if copied > expected_size or copied > MAX_ENTRY_BYTES:
            raise ArchiveError("archive entry expanded beyond its declared bound")
        destination.write(chunk)
    if copied != expected_size:
        raise ArchiveError(
            f"archive entry size mismatch: expected {expected_size}, read {copied}"
        )


def _validate_zip_structure(path: Path) -> None:
    """Bound ZIP metadata before zipfile parses or allocates its entry list."""

    size = path.stat().st_size
    tail_size = min(size, 65_557)
    with path.open("rb") as payload:
        payload.seek(size - tail_size)
        tail = payload.read(tail_size)
    eocd_signature = b"PK\x05\x06"
    candidates: list[tuple[int, tuple]] = []
    for index in range(len(tail)):
        if not tail.startswith(eocd_signature, index) or len(tail) - index < 22:
            continue
        fields = struct.unpack_from("<4s4H2LH", tail, index)
        comment_size = fields[-1]
        absolute = size - tail_size + index
        if absolute + 22 + comment_size == size:
            candidates.append((index, fields))
    if len(candidates) != 1:
        raise ArchiveError("ZIP has missing or ambiguous end-of-central-directory")
    relative_eocd, eocd_fields = candidates[0]
    absolute_eocd = size - tail_size + relative_eocd
    (
        _signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_size,
    ) = eocd_fields
    zip64_locator = False
    if absolute_eocd >= 20:
        with path.open("rb") as payload:
            payload.seek(absolute_eocd - 20)
            zip64_locator = payload.read(4) == b"PK\x06\x07"
    if disk_number != 0 or central_disk != 0 or entries_on_disk != entry_count:
        raise ArchiveError("multidisk ZIP archives are forbidden")
    if (
        entry_count == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or zip64_locator
    ):
        raise ArchiveError("ZIP64 archives are forbidden")
    if entry_count > MAX_ENTRIES:
        raise ArchiveError(f"ZIP contains too many entries: {entry_count}")
    if central_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ArchiveError("ZIP central directory exceeds the metadata bound")
    if central_offset + central_size != absolute_eocd or central_offset > size:
        raise ArchiveError("ZIP central directory is out of range or ambiguous")

    with path.open("rb") as payload:
        payload.seek(central_offset)
        central = payload.read(central_size)
    if len(central) != central_size:
        raise ArchiveError("ZIP central directory is truncated")
    cursor = 0
    parsed_entries = 0
    while cursor < len(central):
        if len(central) - cursor < 46:
            raise ArchiveError("ZIP central directory entry is truncated")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        if fields[0] != b"PK\x01\x02":
            raise ArchiveError("ZIP central directory has an invalid entry signature")
        flag_bits = fields[3]
        compression_method = fields[4]
        if flag_bits & 0x1:
            raise ArchiveError("encrypted ZIP entries are forbidden")
        if compression_method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ArchiveError("ZIP entry uses a forbidden compression method")
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_size, extra_size, entry_comment_size = fields[10:13]
        disk_start = fields[13]
        local_offset = fields[16]
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
            or disk_start == 0xFFFF
        ):
            raise ArchiveError("ZIP64 central-directory fields are forbidden")
        entry_size = 46 + name_size + extra_size + entry_comment_size
        if cursor + entry_size > len(central):
            raise ArchiveError("ZIP central directory variable fields are truncated")
        cursor += entry_size
        parsed_entries += 1
        if parsed_entries > MAX_ENTRIES:
            raise ArchiveError("ZIP central directory exceeds the entry bound")
    if cursor != central_size or parsed_entries != entry_count:
        raise ArchiveError("ZIP central directory entry count is inconsistent")


def _validate_tar_termination(path: Path, logical_end: int) -> None:
    """Require the last payload to be followed only by canonical zero blocks."""

    archive_size = path.stat().st_size
    padding_size = archive_size - logical_end
    if padding_size < 1024 or padding_size % 512 != 0:
        raise ArchiveError("tar.gz has noncanonical end-of-archive padding")
    with path.open("rb") as raw:
        raw.seek(logical_end)
        remaining = padding_size
        while remaining:
            chunk = raw.read(min(1024 * 1024, remaining))
            if not chunk or any(chunk):
                raise ArchiveError("tar.gz has hidden data after its last member")
            remaining -= len(chunk)


def _parse_tar_octal(field: bytes, label: str) -> int:
    if not field or field[0] & 0x80:
        raise ArchiveError(f"tar {label} uses a forbidden numeric encoding")
    value = field.rstrip(b"\x00 ").lstrip(b" ")
    if not value or any(character not in b"01234567" for character in value):
        raise ArchiveError(f"tar {label} is not canonical octal")
    return int(value, 8)


def _validate_raw_tar(path: Path) -> list[tuple[str, int, int]]:
    """Parse bounded canonical regular TAR headers before invoking tarfile."""

    archive_size = path.stat().st_size
    if archive_size % 512 != 0:
        raise ArchiveError("tar stream is not aligned to 512-byte blocks")
    records: list[tuple[str, int, int]] = []
    total = 0
    offset = 0
    with path.open("rb") as raw:
        while True:
            if offset + 512 > archive_size:
                raise ArchiveError("tar stream has no canonical end marker")
            raw.seek(offset)
            header = raw.read(512)
            if len(header) != 512:
                raise ArchiveError("tar header is truncated")
            if not any(header):
                if offset + 1024 > archive_size:
                    raise ArchiveError("tar stream lacks two zero end blocks")
                second_zero = raw.read(512)
                if len(second_zero) != 512 or any(second_zero):
                    raise ArchiveError("tar stream lacks two zero end blocks")
                _validate_tar_termination(path, offset)
                break

            if len(records) >= MAX_ENTRIES:
                raise ArchiveError("tar.gz contains too many entries")
            stored_checksum = _parse_tar_octal(header[148:156], "checksum")
            computed_checksum = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
            if stored_checksum != computed_checksum:
                raise ArchiveError("tar header checksum mismatch")
            if header[257:263] != b"ustar\x00" or header[263:265] != b"00":
                raise ArchiveError("tar header is not canonical ustar")
            if header[156:157] not in {b"\x00", b"0"}:
                raise ArchiveError("tar extensions and special entries are forbidden")
            if any(header[157:257]) or any(header[345:500]):
                raise ArchiveError("tar links and prefixed paths are forbidden")

            name_field = header[:100]
            name_end = name_field.find(b"\x00")
            if name_end < 0 or any(name_field[name_end + 1 :]):
                raise ArchiveError("tar entry name is not canonical NUL-terminated UTF-8")
            try:
                name = _require_plain_name(name_field[:name_end].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ArchiveError("tar entry name is not UTF-8") from error
            size = _parse_tar_octal(header[124:136], "entry size")
            if size > MAX_ENTRY_BYTES:
                raise ArchiveError(f"tar.gz entry {name!r} exceeds the size bound")
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ArchiveError("tar.gz expanded size exceeds the total bound")
            data_offset = offset + 512
            padded_end = data_offset + ((size + 511) // 512) * 512
            if padded_end > archive_size:
                raise ArchiveError("tar entry extends beyond the bounded stream")
            if padded_end > data_offset + size:
                raw.seek(data_offset + size)
                padding = raw.read(padded_end - data_offset - size)
                if any(padding):
                    raise ArchiveError("tar entry has nonzero data padding")
            records.append((name, size, data_offset))
            offset = padded_end
    if not records:
        raise ArchiveError("tar.gz contains no members")
    return records


def _decompress_canonical_gzip(path: Path, raw_path: Path) -> int:
    """Bound a single canonical no-metadata gzip member before TAR parsing."""

    with path.open("rb") as source:
        header = source.read(10)
    if (
        len(header) != 10
        or header[:2] != b"\x1f\x8b"
        or header[2] != 8
        or header[3] != 0
    ):
        raise ArchiveError("tar.gz does not have a canonical metadata-free gzip header")

    hard_limit = MAX_TOTAL_BYTES + MAX_TAR_HEADER_BYTES
    expanded_size = 0
    decompressor = zlib.decompressobj(wbits=31)
    try:
        with path.open("rb") as source, raw_path.open("wb") as raw:
            while chunk := source.read(1024 * 1024):
                if decompressor.eof:
                    raise ArchiveError("tar.gz has concatenated members or trailing bytes")
                pending = chunk
                while pending:
                    output_limit = min(1024 * 1024, hard_limit + 1 - expanded_size)
                    if output_limit < 1:
                        raise ArchiveError("tar.gz expanded stream exceeds the hard bound")
                    output = decompressor.decompress(pending, output_limit)
                    pending = decompressor.unconsumed_tail
                    expanded_size += len(output)
                    if expanded_size > hard_limit:
                        raise ArchiveError("tar.gz expanded stream exceeds the hard bound")
                    raw.write(output)
                    if decompressor.unused_data:
                        raise ArchiveError(
                            "tar.gz has concatenated members or trailing bytes"
                        )
    except zlib.error as error:
        raise ArchiveError("tar.gz gzip member failed checksum or stream validation") from error
    if not decompressor.eof or decompressor.unused_data:
        raise ArchiveError("tar.gz gzip member is truncated or has trailing bytes")
    return expanded_size


def safe_extract_zip(path: Path, destination: Path, expected: list[str]) -> list[Path]:
    """Extract an exact flat ZIP of regular files after hostile-input checks."""

    archive_size = path.stat().st_size
    if archive_size < 1 or archive_size > MAX_ARCHIVE_BYTES:
        raise ArchiveError(f"ZIP size {archive_size} is outside the allowed range")
    _validate_zip_structure(path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ENTRIES:
            raise ArchiveError(f"ZIP contains too many entries: {len(infos)}")
        names: list[str] = []
        total = 0
        for info in infos:
            name = _require_plain_name(info.filename)
            mode = (info.external_attr >> 16) & 0o177777
            kind = stat.S_IFMT(mode)
            if info.is_dir() or kind == stat.S_IFLNK or kind not in {0, stat.S_IFREG}:
                raise ArchiveError(f"ZIP entry {name!r} is not a regular file")
            if info.file_size < 0 or info.file_size > MAX_ENTRY_BYTES:
                raise ArchiveError(f"ZIP entry {name!r} exceeds the size bound")
            if info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1):
                raise ArchiveError(f"ZIP entry {name!r} exceeds the compression ratio bound")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ArchiveError("ZIP expanded size exceeds the total bound")
            names.append(name)
        _require_expected(names, expected)
        extracted: list[Path] = []
        for info in infos:
            output = destination / info.filename
            with archive.open(info, "r") as source, output.open("xb") as target:
                _copy_bounded(source, target, info.file_size)
            extracted.append(output)
        return extracted


def safe_extract_tar_gz(
    path: Path, destination: Path, expected: list[str]
) -> list[Path]:
    """Extract an exact flat gzip-compressed TAR of regular files."""

    archive_size = path.stat().st_size
    if archive_size < 1 or archive_size > MAX_ARCHIVE_BYTES:
        raise ArchiveError(f"tar.gz size {archive_size} is outside the allowed range")
    raw_handle, raw_name = tempfile.mkstemp(prefix="wenlan-release-tar-", suffix=".tar")
    os.close(raw_handle)
    raw_path = Path(raw_name)
    try:
        expanded_size = _decompress_canonical_gzip(path, raw_path)
        if expanded_size > MAX_COMPRESSION_RATIO * max(archive_size, 1):
            raise ArchiveError("tar.gz exceeds the compression ratio bound")
        raw_records = _validate_raw_tar(raw_path)
        with tarfile.open(raw_path, mode="r:") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ENTRIES:
                raise ArchiveError(f"tar.gz contains too many entries: {len(members)}")
            names: list[str] = []
            total = 0
            for member in members:
                name = _require_plain_name(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise ArchiveError(f"tar.gz entry {name!r} is not a regular file")
                if member.size < 0 or member.size > MAX_ENTRY_BYTES:
                    raise ArchiveError(f"tar.gz entry {name!r} exceeds the size bound")
                total += member.size
                if total > MAX_TOTAL_BYTES:
                    raise ArchiveError("tar.gz expanded size exceeds the total bound")
                names.append(name)
            _require_expected(names, expected)
            observed = [
                (member.name, member.size, member.offset_data) for member in members
            ]
            if observed != raw_records:
                raise ArchiveError("tarfile interpretation differs from bounded preflight")
            destination.mkdir(parents=True, exist_ok=False)
            extracted: list[Path] = []
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveError(f"tar.gz entry {member.name!r} has no readable payload")
                output = destination / member.name
                with source, output.open("xb") as target:
                    _copy_bounded(source, target, member.size)
                extracted.append(output)
            return extracted
    finally:
        raw_path.unlink(missing_ok=True)


def safe_extract_archive(
    path: Path, destination: Path, archive_type: str, expected: list[str]
) -> list[Path]:
    if archive_type == "zip":
        return safe_extract_zip(path, destination, expected)
    if archive_type == "tar.gz":
        return safe_extract_tar_gz(path, destination, expected)
    raise ArchiveError(f"unsupported archive type {archive_type!r}")


def _require_source(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArchiveError(f"release source is not a regular file: {path}")
    size = path.stat().st_size
    if size < 1 or size > MAX_ENTRY_BYTES:
        raise ArchiveError(f"release source size is outside the allowed range: {path}")


def create_tar_gz(
    output: Path, sources: list[tuple[Path, str]], source_date_epoch: int
) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw,
            mtime=source_date_epoch,
        ) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for source, archive_name in sources:
                    _require_source(source)
                    name = _require_plain_name(archive_name)
                    info = tarfile.TarInfo(name)
                    info.size = source.stat().st_size
                    info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
                    info.mtime = source_date_epoch
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as payload:
                        archive.addfile(info, payload)
    temporary.replace(output)


def create_zip(
    output: Path, sources: list[tuple[Path, str]], source_date_epoch: int
) -> None:
    # ZIP timestamps cannot represent dates before 1980.
    timestamp = max(source_date_epoch, 315532800)
    date_time = tuple(time.gmtime(timestamp)[:6])
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(
        temporary, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source, archive_name in sources:
            _require_source(source)
            name = _require_plain_name(archive_name)
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if source.suffix.lower() == ".exe" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            with source.open("rb") as payload, archive.open(info, "w") as target:
                shutil.copyfileobj(payload, target, length=1024 * 1024)
    temporary.replace(output)
