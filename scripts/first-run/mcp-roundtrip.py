#!/usr/bin/env python3
"""First-run gauntlet: drive `wenlan-mcp` over stdio JSON-RPC.

initialize -> tools/list -> capture -> recall (polled up to 60s) [-> brief].
Lifted from scripts/smoke-mcp.sh. A failed step never raises: every step
appends a row to $GAUNTLET_OUT/findings.tsv (same TSV shape as lib.sh:
channel, name, status, rc, detail) and writes its raw response to
$GAUNTLET_OUT/checks/<name>.log. Always exits 0 — the channel script's
`evaluate` turns FAIL rows into a nonzero exit.

Env:
  MCP_BIN               path to wenlan-mcp (required)
  MCP_ARGS              JSON list of extra args (default [])
  EXPECT_TOOL_COUNT     when set, tools/list must return exactly this many tools
  MCP_TOOLS             comma list of tools that must be advertised (default
                        "capture,recall"); when it names "brief", brief is also
                        called with {} and must not error
  GAUNTLET_OUT          default ./gauntlet-out
  GAUNTLET_CHANNEL      default "mcp"
  WENLAN_MCP_CACHE_DIR  default $GAUNTLET_OUT/mcp-cache (keeps the self-update
                        probe out of the user cache)
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

SENTINEL = "walrus-observatory-5507"
OUT = Path(os.environ.get("GAUNTLET_OUT") or Path.cwd() / "gauntlet-out")
CHANNEL = os.environ.get("GAUNTLET_CHANNEL") or "mcp"
CHECKS = OUT / "checks"
TSV = OUT / "findings.tsv"
CHECKS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WENLAN_MCP_CACHE_DIR", str(OUT / "mcp-cache"))
Path(os.environ["WENLAN_MCP_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def _escape(detail):
    # One line, no tabs, capped — mirrors _gauntlet_escape in lib.sh.
    return detail.replace("\t", " ").replace("\r", " ").replace("\n", "|")[:2000]


def record(status, name, rc, detail=""):
    with TSV.open("a", encoding="utf-8") as fh:
        fh.write(f"{CHANNEL}\t{name}\t{status}\t{rc}\t{_escape(detail)}\n")
    short = _escape(detail)[:200]
    print(f"[{status}] {name} (rc={rc})" + (f" — {short}" if short else ""), flush=True)


def log(name, text):
    (CHECKS / f"{name}.log").write_text(text, encoding="utf-8")


class StepFailed(Exception):
    pass


class Transport(StepFailed):
    """The server died or stopped answering: later steps cannot run."""


# ---------------------------------------------------------------------------

if not os.environ.get("MCP_BIN"):
    record("FAIL", "mcp-initialize", 2, "MCP_BIN (path to wenlan-mcp) is required")
    sys.exit(0)

try:
    extra_args = json.loads(os.environ.get("MCP_ARGS") or "[]")
    if not isinstance(extra_args, list):
        raise ValueError("MCP_ARGS must be a JSON list")
except ValueError as exc:
    record("FAIL", "mcp-initialize", 2, f"bad MCP_ARGS: {exc}")
    sys.exit(0)

required_tools = [t.strip() for t in (os.environ.get("MCP_TOOLS") or "capture,recall").split(",") if t.strip()]
expect_count = os.environ.get("EXPECT_TOOL_COUNT")

stderr_log = (CHECKS / "mcp-server-stderr.log").open("wb")
try:
    proc = subprocess.Popen(
        [os.environ["MCP_BIN"], *extra_args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_log,
    )
except OSError as exc:
    record("FAIL", "mcp-initialize", 127, f"cannot start {os.environ['MCP_BIN']}: {exc}")
    sys.exit(0)

buf = b""
next_id = 0
dead = None  # reason string once the transport is gone

# A pump thread hands stdout chunks over a queue: select() only accepts
# sockets on Windows (WSAENOTSOCK/WSANOTINITIALISED on a pipe), and a queue
# read with a timeout is the same wait on every platform. An empty chunk
# means the server closed stdout.
_chunks = queue.Queue()


def _pump():
    fd = proc.stdout.fileno()
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            chunk = b""
        _chunks.put(chunk)
        if not chunk:
            return


threading.Thread(target=_pump, name="stdout-pump", daemon=True).start()


def send(obj):
    try:
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise Transport(f"wenlan-mcp closed stdin: {exc}")


def recv(want_id, timeout=60):
    # Frame by hand from raw chunks: a buffered readline() can hang on a
    # partial frame or strand a second frame already sitting in Python's
    # buffer while the caller waits for more bytes.
    global buf
    deadline = time.monotonic() + timeout
    while True:
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                return msg
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Transport(f"timeout waiting for response id={want_id}")
        try:
            chunk = _chunks.get(timeout=min(remaining, 1))
        except queue.Empty:
            continue
        if not chunk:
            raise Transport("wenlan-mcp closed stdout")
        buf += chunk


def call(method, params=None, timeout=60):
    global next_id
    next_id += 1
    req = {"jsonrpc": "2.0", "id": next_id, "method": method}
    if params is not None:
        req["params"] = params
    send(req)
    return recv(next_id, timeout=timeout)


def expect_ok(msg, what):
    if "error" in msg:
        raise StepFailed(f"{what} returned JSON-RPC error: {json.dumps(msg['error'])}")
    if msg.get("result", {}).get("isError"):
        raise StepFailed(f"{what} returned tool error: {json.dumps(msg['result'])}")
    return msg["result"]


def step(name, fn):
    """Run fn(); record PASS/FAIL; return its value or None. Never raises."""
    global dead
    if dead:
        record("FAIL", name, 1, f"not run: {dead}")
        return None
    try:
        value, detail = fn()
    except Transport as exc:
        dead = f"transport failed at {name}: {exc}"
        log(name, str(exc))
        record("FAIL", name, 1, str(exc))
        return None
    except StepFailed as exc:
        log(name, str(exc))
        record("FAIL", name, 1, str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 — a driver bug must still be a row, not a crash
        log(name, repr(exc))
        record("FAIL", name, 1, f"driver error: {exc!r}")
        return None
    record("PASS", name, 0, detail)
    return value


# ---------------------------------------------------------------------------

def do_initialize():
    msg = call("initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "first-run-gauntlet", "version": "0"}})
    log("mcp-initialize", json.dumps(msg, indent=2))
    init = expect_ok(msg, "initialize")
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    server = init.get("serverInfo", {})
    return init, f"server {server.get('name', '?')} {server.get('version', '')}".strip()


def do_tools_list():
    msg = call("tools/list")
    log("mcp-tools-list", json.dumps(msg, indent=2))
    tools = sorted(t["name"] for t in expect_ok(msg, "tools/list")["tools"])
    record("INFO", "mcp-tool-count", 0, f"{len(tools)}: {', '.join(tools)}")
    problems = []
    missing = [t for t in required_tools if t not in tools]
    if missing:
        problems.append(f"missing required tool(s) {missing}")
    if expect_count is not None and len(tools) != int(expect_count):
        problems.append(f"expected {expect_count} tools, got {len(tools)}")
    if problems:
        raise StepFailed("; ".join(problems) + f" (advertised: {tools})")
    return tools, f"{len(tools)} tools"


def do_capture():
    msg = call("tools/call", {
        "name": "capture",
        "arguments": {"content": f"The {SENTINEL} sentinel lives in the MCP smoke."}})
    log("mcp-capture", json.dumps(msg, indent=2))
    expect_ok(msg, "capture")
    return True, "captured sentinel"


def do_recall():
    # One monotonic window for the whole poll — per-retry timeouts would let
    # the advertised 60s stretch to 30 minutes against a hung daemon.
    poll_deadline = time.monotonic() + 60
    polls = 0
    last = None
    while time.monotonic() < poll_deadline:
        polls += 1
        last = call("tools/call", {
            "name": "recall",
            "arguments": {"query": "walrus observatory sentinel"}},
            timeout=max(1, poll_deadline - time.monotonic()))
        log("mcp-recall", json.dumps(last, indent=2))
        result = expect_ok(last, "recall")
        if SENTINEL in json.dumps(result):
            return True, f"sentinel round-tripped after {polls} poll(s)"
        time.sleep(2)
    raise StepFailed(f"captured sentinel not present in recall output within 60s ({polls} polls); last: {json.dumps(last)}")


def do_brief():
    msg = call("tools/call", {"name": "brief", "arguments": {}})
    log("mcp-brief", json.dumps(msg, indent=2))
    result = expect_ok(msg, "brief")
    return True, f"brief ok ({len(json.dumps(result))} bytes)"


step("mcp-initialize", do_initialize)
tools = step("mcp-tools-list", do_tools_list)
step("mcp-capture", do_capture)
step("mcp-recall", do_recall)
if "brief" in required_tools:
    if tools is not None and "brief" not in tools:
        record("FAIL", "mcp-brief", 1, "not run: brief is not advertised by tools/list")
    else:
        step("mcp-brief", do_brief)

try:
    proc.stdin.close()
except OSError:
    pass
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
stderr_log.close()
sys.exit(0)
