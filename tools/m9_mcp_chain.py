"""M9 MCP chain: end-to-end acceptance for the DepthShaderKit integration.

Builds the app (Debug, known build dir), launches it headlessly-ish (GUI app with
built-in MCP server on 127.0.0.1:8378), then drives a scripted chain over JSON-RPC:

  load_image (design/depth_planes.png, real CoreML inference)
  -> export gray16 baseline
  -> transform_depth (rotate+scale+z_shift, then translate) -> fuse_depth
  -> export gray16 edited
  -> edit_reset -> same chain again -> export gray16 edited2

Gates:
  - load_image succeeds (real inference, generous timeout for model load)
  - exports exist, PNG dims == input image dims (export upscales to input res)
  - edited PNG differs from baseline (edits actually applied)
  - edited2 byte-identical to edited (determinism)
  - transform checksum after edit_reset matches the first run's checksum
    (UI and MCP share the single AppState edit path, so identical MCP results
    are the automated proxy for UI-vs-MCP pixel identity)

Exits 0 on M9 PASS, 1 on any failure. The app process is always killed.
"""
import http.client
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XCODE_DEVELOPER = "/Applications/Xcode.app/Contents/Developer"
BUILD_DIR = "/tmp/da3depth-m9-build"
APP_BINARY = os.path.join(BUILD_DIR, "DA3Depth.app", "Contents", "MacOS", "DA3Depth")
IMAGE = os.path.join(ROOT, "design", "depth_planes.png")
# Own port (not the default 8378) so the chain can run alongside a user-launched app.
PORT = int(os.environ.get("M9_PORT", "8379"))

failures = []


def gate(ok, label):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


def rpc(method, params=None, timeout=60):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": params or {}}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    conn.request("POST", "/mcp", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return json.loads(data)["result"]


def tool(name, args=None, timeout=60):
    result = rpc("tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
    text = result["content"][0]["text"]
    print(f"  {name}: {text}")
    return text


def png_dims(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR"
    return struct.unpack(">II", head[16:24])


def checksum_of(text):
    m = re.search(r"checksum ([0-9.eE+-]+)", text)
    return m.group(1) if m else None


def main():
    if not os.path.exists(IMAGE):
        print(f"M9 FAIL (test image missing: {IMAGE})")
        sys.exit(1)

    print("building app...")
    env = dict(os.environ, DEVELOPER_DIR=XCODE_DEVELOPER)
    r = subprocess.run(
        ["xcodebuild", "-project", os.path.join(ROOT, "DA3Depth.xcodeproj"),
         "-scheme", "DA3Depth", "-configuration", "Debug",
         f"CONFIGURATION_BUILD_DIR={BUILD_DIR}", "build"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:], file=sys.stderr)
        print("M9 FAIL (xcodebuild error)")
        sys.exit(1)

    # port must be free (another app instance would confuse the chain)
    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", PORT))
        print(f"M9 FAIL (port {PORT} already in use — quit the other DA3Depth instance)")
        sys.exit(1)
    except OSError:
        pass
    finally:
        probe.close()

    app = subprocess.Popen([APP_BINARY], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env=dict(os.environ, DA3DEPTH_MCP_PORT=str(PORT)))
    try:
        deadline = time.time() + 60
        while True:
            try:
                rpc("initialize")
                break
            except OSError:
                if time.time() > deadline:
                    print("M9 FAIL (MCP server did not come up on 8378)")
                    sys.exit(1)
                time.sleep(0.5)
        print("app launched, MCP server up")

        baseline = "/tmp/m9_baseline.png"
        edited1 = "/tmp/m9_edited1.png"
        edited2 = "/tmp/m9_edited2.png"

        print("chain: load -> baseline export -> transform -> fuse -> export")
        t = tool("load_image", {"path": IMAGE}, timeout=300)
        gate("完成" in t, "load_image + CoreML inference")
        tool("export", {"kind": "gray16", "path": baseline})

        r1 = tool("transform_depth", {"rotate_deg": 12, "scale": 1.1, "z_shift": 0.15})
        cs1 = checksum_of(r1)
        tool("transform_depth", {"tx": 15, "ty": -8})
        tool("fuse_depth", {"layers": [0]})
        tool("export", {"kind": "gray16", "path": edited1})

        print("chain: edit_reset -> identical transforms -> fuse -> export")
        tool("edit_reset")
        r1b = tool("transform_depth", {"rotate_deg": 12, "scale": 1.1, "z_shift": 0.15})
        cs1b = checksum_of(r1b)
        tool("transform_depth", {"tx": 15, "ty": -8})
        tool("fuse_depth", {"layers": [0]})
        tool("export", {"kind": "gray16", "path": edited2})

        in_dims = png_dims(IMAGE)
        gate(os.path.exists(baseline) and png_dims(baseline) == in_dims,
             f"baseline export dims == input {in_dims}")
        gate(os.path.exists(edited1) and png_dims(edited1) == in_dims,
             "edited export dims == input")
        with open(baseline, "rb") as f:
            base_bytes = f.read()
        with open(edited1, "rb") as f:
            edit1_bytes = f.read()
        with open(edited2, "rb") as f:
            edit2_bytes = f.read()
        gate(edit1_bytes != base_bytes, "edited export differs from baseline (edits applied)")
        gate(edit1_bytes == edit2_bytes, "repeat chain export byte-identical (determinism)")
        gate(cs1 is not None and cs1 == cs1b,
             f"transform checksum stable across edit_reset ({cs1} == {cs1b})")
    finally:
        app.terminate()
        try:
            app.wait(timeout=10)
        except subprocess.TimeoutExpired:
            app.kill()

    if failures:
        print(f"M9 FAIL ({len(failures)} gates failed)")
        sys.exit(1)
    print("M9 PASS (UI 与 MCP 共用同一 AppState 编辑路径；MCP 端逐像素一致即为 UI 一致的自动化代理)")
    sys.exit(0)


if __name__ == "__main__":
    main()
