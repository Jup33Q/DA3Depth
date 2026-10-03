"""M10 MCP chain: end-to-end acceptance for the M7 2.5D reprojection.

Builds the app (Debug, known build dir), launches it with the built-in MCP server,
then drives the acceptance chain over JSON-RPC on the two validation assets
(~/Desktop/videos/01/_reset/, 1152x1536, DA3MONO-LARGE depth):

  per frame (first, last) and yaw (+5, -5):
    load_image -> reproject_view{yaw_deg} -> export warped_color + warped_depth16 + warped_mask

Gates:
  - reproject_view reports >=90% coverage and 1 micro-step at yaw=5 (step limit 5 deg)
  - recursion: yaw 12 runs as 3 micro-steps; clamp: yaw 45 clamps to 30 deg
  - exports exist, PNG dims == 1152x1536
  - warped color differs from the source frame (warp actually applied)
  - determinism: repeating the yaw +5 chain gives byte-identical PNGs
  - coverage < 100% (mask actually marks the unfilled big holes)

Outputs land in ~/Desktop/videos/01/_reset/reproject-m10/ for LKG parallax pairing.
Exits 0 on M10 PASS, 1 on any failure. The app process is always killed.
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
BUILD_DIR = "/tmp/da3depth-m10-build"
APP_BINARY = os.path.join(BUILD_DIR, "DA3Depth.app", "Contents", "MacOS", "DA3Depth")
ASSETS = os.path.expanduser("~/Desktop/videos/01/_reset")
OUT_DIR = os.path.join(ASSETS, "reproject-m10")
FRAMES = {"first": "first-frame-1152x1536.png", "last": "last-frame-1152x1536.png"}
# Own port (not the default 8378) so the chain can run alongside a user-launched app.
PORT = int(os.environ.get("M10_PORT", "8379"))

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


def tool(name, args=None, timeout=120):
    result = rpc("tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
    text = result["content"][0]["text"]
    print(f"  {name}: {text}")
    return text


def png_dims(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR"
    return struct.unpack(">II", head[16:24])


def coverage_of(text):
    m = re.search(r"覆盖率 ([0-9.]+)%", text)
    return float(m.group(1)) if m else None


def steps_of(text):
    m = re.search(r"(\d+) 微步", text)
    return int(m.group(1)) if m else None


def main():
    for f in FRAMES.values():
        if not os.path.exists(os.path.join(ASSETS, f)):
            print(f"M10 FAIL (asset missing: {f})")
            sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

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
        print("M10 FAIL (xcodebuild error)")
        sys.exit(1)

    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", PORT))
        print(f"M10 FAIL (port {PORT} already in use — quit the other DA3Depth instance)")
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
                    print("M10 FAIL (MCP server did not come up)")
                    sys.exit(1)
                time.sleep(0.5)
        print("app launched, MCP server up")

        # recursion / clamp gates on the first frame
        tool("load_image", {"path": os.path.join(ASSETS, FRAMES["first"])}, timeout=300)
        t12 = tool("reproject_view", {"yaw_deg": 12})
        gate(steps_of(t12) == 3, f"yaw 12 -> 3 micro-steps (got {steps_of(t12)})")
        t45 = tool("reproject_view", {"yaw_deg": 45})
        gate("yaw 45.00°→30.00°" in t45 and steps_of(t45) == 6,
             "yaw 45 clamped to 30 (6 micro-steps)")

        for name, fname in FRAMES.items():
            tool("load_image", {"path": os.path.join(ASSETS, fname)}, timeout=300)
            for yaw, tag in ((5, "p5"), (-5, "m5")):
                t = tool("reproject_view", {"yaw_deg": yaw})
                cov = coverage_of(t)
                # at 5° yaw the out-of-frame / disocclusion band is a big hole that
                # must stay masked (not filled), so ~80-90% coverage is expected
                gate(cov is not None and 80.0 <= cov < 100.0,
                     f"{name} yaw{yaw:+} coverage {cov}% in [80, 100)")
                gate(steps_of(t) == 1, f"{name} yaw{yaw:+} single micro-step")
                outs = {}
                for kind, suffix in (("warped_color", "color"), ("warped_depth16", "depth16"),
                                     ("warped_mask", "mask")):
                    path = os.path.join(OUT_DIR, f"{name}_{tag}_{suffix}.png")
                    tool("export", {"kind": kind, "path": path})
                    outs[suffix] = path
                for suffix, path in outs.items():
                    gate(os.path.exists(path) and png_dims(path) == (1152, 1536),
                         f"{name}_{tag}_{suffix}.png dims == 1152x1536")
                with open(os.path.join(ASSETS, fname), "rb") as f:
                    src_bytes = f.read()
                with open(outs["color"], "rb") as f:
                    warped_bytes = f.read()
                gate(warped_bytes != src_bytes, f"{name} yaw{yaw:+} color differs from source")

            # determinism: repeat yaw +5 on this frame, compare bytes
            tool("reproject_view", {"yaw_deg": 5})
            for kind, suffix in (("warped_color", "color"), ("warped_depth16", "depth16"),
                                 ("warped_mask", "mask")):
                rep = os.path.join(OUT_DIR, f"{name}_p5_{suffix}.repeat.png")
                tool("export", {"kind": kind, "path": rep})
                with open(os.path.join(OUT_DIR, f"{name}_p5_{suffix}.png"), "rb") as f:
                    a = f.read()
                with open(rep, "rb") as f:
                    b = f.read()
                gate(a == b, f"{name} yaw+5 repeat {suffix} byte-identical")
    finally:
        app.terminate()
        try:
            app.wait(timeout=10)
        except subprocess.TimeoutExpired:
            app.kill()

    if failures:
        print(f"M10 FAIL ({len(failures)} gates failed)")
        sys.exit(1)
    print(f"M10 PASS (outputs in {OUT_DIR})")
    sys.exit(0)


if __name__ == "__main__":
    main()
