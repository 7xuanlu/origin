#!/usr/bin/env python3
"""Build and smoke a runtime image from an exact validated release archive.

This command deliberately stops before registry authentication or push. A
release workflow must invoke it before docker/login-action so the candidate
binary is never executed on a runner carrying GHCR credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from release_archive import ArchiveError, safe_extract_archive


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_validator():
    path = SCRIPT_DIR / "validate-release-candidate.py"
    spec = importlib.util.spec_from_file_location("runtime_release_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validate-release-candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()
TARGETS = {
    "x86_64-unknown-linux-gnu": {
        "archive": "wenlan-linux-x64.tar.gz",
        "architecture": "amd64",
        "platform": "linux/amd64",
    },
    "aarch64-unknown-linux-gnu": {
        "archive": "wenlan-linux-arm64.tar.gz",
        "architecture": "arm64",
        "platform": "linux/arm64",
    },
}
ARCHIVE_MEMBERS = ["wenlan", "wenlan-server", "wenlan-mcp"]
REQUIRED_ENV = {
    "WENLAN_BIND_ADDR=0.0.0.0:7878",
    "WENLAN_PORT=7878",
    "WENLAN_DATA_DIR=/data",
}


class RuntimeImageError(RuntimeError):
    """A runtime image differs from its validated release payload or contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload:
        while chunk := payload.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str], *, timeout: int = 120, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        output = (result.stdout or "").strip()[-4000:]
        raise RuntimeImageError(
            f"command failed ({result.returncode}): {command[0]} {command[1]}\n{output}"
        )
    return result


def _asset_claim(receipt: dict, target: str) -> tuple[dict, dict]:
    target_entry = TARGETS.get(target)
    if target_entry is None:
        raise RuntimeImageError(f"unsupported runtime target {target!r}")
    matches = [
        item
        for item in receipt.get("assets", [])
        if isinstance(item, dict) and item.get("name") == target_entry["archive"]
    ]
    if len(matches) != 1:
        raise RuntimeImageError("closed receipt does not contain one exact Linux asset")
    return target_entry, matches[0]


def prepare_context(
    receipt: dict, assets_dir: Path, target: str, context_dir: Path
) -> dict[str, str | int]:
    target_entry, claim = _asset_claim(receipt, target)
    archive = assets_dir / target_entry["archive"]
    if not archive.is_file() or archive.is_symlink():
        raise RuntimeImageError(f"validated archive is not a regular file: {archive.name}")
    actual_size = archive.stat().st_size
    actual_archive_sha = sha256_file(archive)
    if actual_size != claim.get("size") or actual_archive_sha != claim.get("sha256"):
        raise RuntimeImageError("Linux archive bytes differ from the closed receipt")

    extracted = context_dir / "extracted"
    payloads = safe_extract_archive(archive, extracted, "tar.gz", ARCHIVE_MEMBERS)
    by_name = {path.name: path for path in payloads}
    if set(by_name) != set(ARCHIVE_MEMBERS):
        raise RuntimeImageError("Linux archive member inventory is not exact")
    server = by_name["wenlan-server"]
    server_sha = sha256_file(server)

    build_context = context_dir / "context"
    data_dir = build_context / "data"
    data_dir.mkdir(parents=True)
    # Keep the directory in BuildKit's context tar so COPY can materialize it
    # with numeric nonroot ownership. This file is harmless inside /data.
    (data_dir / ".volume-seed").write_bytes(b"")
    destination = build_context / "wenlan-server"
    shutil.copyfile(server, destination)
    destination.chmod(0o755)
    if sha256_file(destination) != server_sha:
        raise RuntimeImageError("prepared server bytes differ from the extracted member")
    return {
        "archive": str(target_entry["archive"]),
        "archive_sha256": actual_archive_sha,
        "archive_size": actual_size,
        "architecture": str(target_entry["architecture"]),
        "platform": str(target_entry["platform"]),
        "server_sha256": server_sha,
        "build_context": str(build_context),
    }


def _inspect_image(image: str, architecture: str) -> dict:
    result = _run(["docker", "image", "inspect", image], timeout=30)
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeImageError("docker image inspect returned invalid JSON") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RuntimeImageError("docker image inspect returned an unexpected shape")
    image_info = values[0]
    config = image_info.get("Config")
    if not isinstance(config, dict):
        raise RuntimeImageError("runtime image has no config")
    if image_info.get("Os") != "linux" or image_info.get("Architecture") != architecture:
        raise RuntimeImageError("runtime image OS or architecture is wrong")
    if config.get("User") != "65532:65532":
        raise RuntimeImageError("runtime image must use numeric nonroot uid/gid 65532")
    if config.get("Entrypoint") != ["/usr/local/bin/wenlan-server"]:
        raise RuntimeImageError("runtime image entrypoint is not exact")
    volumes = config.get("Volumes")
    if not isinstance(volumes, dict) or set(volumes) != {"/data"}:
        raise RuntimeImageError("runtime image volume inventory is not exact")
    environment = config.get("Env")
    if not isinstance(environment, list) or not REQUIRED_ENV.issubset(set(environment)):
        raise RuntimeImageError("runtime image environment contract is incomplete")
    image_id = image_info.get("Id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeImageError("runtime image has no content-addressed image id")
    return image_info


def _remove_container(name: str) -> None:
    _run(["docker", "rm", "-f", "-v", name], timeout=30, check=False)


def _verify_copied_binary(image: str, expected_sha: str, root: Path) -> None:
    name = f"wenlan-runtime-copy-{uuid.uuid4().hex[:12]}"
    copied = root / "image-wenlan-server"
    try:
        _run(["docker", "create", "--name", name, image], timeout=30)
        _run(
            ["docker", "cp", f"{name}:/usr/local/bin/wenlan-server", str(copied)],
            timeout=60,
        )
        if sha256_file(copied) != expected_sha:
            raise RuntimeImageError("image server bytes differ from the validated archive")
    finally:
        _remove_container(name)


def _request(url: str, *, body: dict | None = None, timeout: int = 3) -> bytes:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeImageError(f"runtime smoke returned HTTP {response.status}")
        return response.read(2 * 1024 * 1024)


def _semantic_smoke(image: str) -> None:
    name = f"wenlan-runtime-smoke-{uuid.uuid4().hex[:12]}"
    try:
        _run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                name,
                "--publish",
                "127.0.0.1::7878",
                image,
            ],
            timeout=30,
        )
        port_result = _run(["docker", "port", name, "7878/tcp"], timeout=30)
        binding = port_result.stdout.strip().splitlines()[0]
        try:
            port = int(binding.rsplit(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeImageError("docker returned an invalid published port") from error
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        while True:
            try:
                _request(f"{base}/api/health")
                break
            except (OSError, urllib.error.URLError, RuntimeImageError):
                if time.monotonic() >= deadline:
                    logs = _run(["docker", "logs", name], timeout=30, check=False).stdout
                    raise RuntimeImageError(
                        f"runtime image did not become healthy\n{(logs or '')[-4000:]}"
                    )
                time.sleep(1)
        sentinel = "release runtime image smoke memory"
        _request(
            f"{base}/api/memory/store",
            body={
                "content": sentinel,
                "memory_type": "lesson",
            },
            timeout=120,
        )
        search = _request(
            f"{base}/api/memory/search",
            body={"query": "release runtime image smoke", "limit": 3},
            timeout=30,
        )
        if sentinel.encode("utf-8") not in search:
            raise RuntimeImageError("runtime image search did not return the stored sentinel")
        _request(f"{base}/api/status")
    finally:
        _remove_container(name)


def verify_runtime_image(
    receipt_path: Path,
    assets_dir: Path,
    target: str,
    image: str,
    dockerfile: Path,
) -> dict:
    receipt = VALIDATOR.read_closed_receipt(receipt_path)
    if not dockerfile.is_file() or dockerfile.is_symlink():
        raise RuntimeImageError("release runtime Dockerfile is missing or not regular")
    with tempfile.TemporaryDirectory(prefix="wenlan-release-runtime-") as raw:
        root = Path(raw)
        evidence = prepare_context(receipt, assets_dir, target, root)
        _run(
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                str(evidence["platform"]),
                "--load",
                "--file",
                str(dockerfile.resolve()),
                "--tag",
                image,
                str(evidence["build_context"]),
            ],
            timeout=600,
            capture=False,
        )
        image_info = _inspect_image(image, str(evidence["architecture"]))
        _verify_copied_binary(image, str(evidence["server_sha256"]), root)
        _semantic_smoke(image)
        return {
            key: value for key, value in evidence.items() if key != "build_context"
        } | {"image": image, "image_id": image_info["Id"], "status": "passed"}


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--image", required=True)
    parser.add_argument("--dockerfile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        raise RuntimeImageError("output must be a new file in an existing directory")
    result = verify_runtime_image(
        args.receipt, args.assets_dir, args.target, args.image, args.dockerfile
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (
        ArchiveError,
        RuntimeImageError,
        OSError,
        subprocess.SubprocessError,
        urllib.error.URLError,
    ) as error:
        print(f"verify-release-runtime-image: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
