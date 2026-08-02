#!/usr/bin/env python3
"""Create the exact six-asset Wenlan release archive surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from release_archive import ArchiveError, create_tar_gz, create_zip
from release_targets import TargetError, release_assets, require_target


def package_target(
    target: str,
    binary_dir: Path,
    output_dir: Path,
    source_date_epoch: int,
) -> list[Path]:
    require_target(target)
    if source_date_epoch < 0 or source_date_epoch > 4_102_444_800:
        raise ArchiveError("source-date-epoch is outside the supported range")
    if not binary_dir.is_dir():
        raise ArchiveError(f"binary directory does not exist: {binary_dir}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ArchiveError(f"output directory must be absent or empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    outputs: list[Path] = []
    for asset in release_assets(target=target):
        sources = [(binary_dir / member, member) for member in asset["members"]]
        output = output_dir / asset["name"]
        if asset["archive"] == "tar.gz":
            create_tar_gz(output, sources, source_date_epoch)
        elif asset["archive"] == "zip":
            create_zip(output, sources, source_date_epoch)
        else:
            raise ArchiveError(f"unsupported canonical archive {asset['archive']!r}")
        outputs.append(output)
    return outputs


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    args = parser.parse_args(argv)
    outputs = package_target(
        args.target,
        args.binary_dir,
        args.output_dir,
        args.source_date_epoch,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (ArchiveError, TargetError, OSError) as error:
        print(f"package-release-artifacts: {error}", file=sys.stderr)
        raise SystemExit(1)
