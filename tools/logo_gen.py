"""Logo generation for DA3Depth via flux-klein-studio MCP backend (stdio).

Spawns `swift run FluxKleinStudio mcp` and issues generate_image calls.
Writes candidates to /tmp/da3_logo/.
"""
import json, os, subprocess, sys, time

ROOT = os.path.expanduser("~/Documents/kimi/workspace/flux-klein-studio")
OUT = "/tmp/da3_logo"
os.makedirs(OUT, exist_ok=True)

env = dict(os.environ, DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
BIN = os.path.join(ROOT, ".build/arm64-apple-macosx/release/FluxKleinStudio")
proc = subprocess.Popen([BIN, "mcp"],
                        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)

def send(msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

def call(req_id, name, args):
    send({"jsonrpc": "2.0", "id": req_id, "method": "tools/call",
          "params": {"name": name, "arguments": args}})
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed stdout")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            continue
        if resp.get("id") == req_id:
            return resp

send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
while True:
    if json.loads(proc.stdout.readline()).get("id") == 0:
        break
print("MCP initialized", flush=True)

STYLE = ("flat minimalist macOS app icon, dark charcoal rounded-square background, "
         "inferno depth-colormap gradient (deep purple, magenta, orange, warm yellow), "
         "crisp clean vector style, high contrast, centered composition, generous margins, "
         "no text, no watermark")
JOBS = [
    ("d_layers", STYLE + ", a bold geometric capital letter D formed by stacked horizontal "
     "layered planes receding into depth like a depth map", 11),
    ("contour_rings", STYLE + ", concentric topographic depth contour rings on a smooth 3D "
     "surface seen from above, brightest ring at the center", 22),
    ("depth_planes", STYLE + ", three translucent horizontal glass planes floating at "
     "different depths in subtle perspective, glowing gradient edges, soft depth fog", 33),
]

for i, (name, prompt, seed) in enumerate(JOBS, start=1):
    path = f"{OUT}/{name}.png"
    t0 = time.time()
    resp = call(i, "generate_image", {"prompt": prompt, "width": 1024, "height": 1024,
                                      "steps": 6, "guidance": 1.0, "seed": seed,
                                      "output_path": path})
    dt = time.time() - t0
    err = resp.get("result", {}).get("isError", False)
    text = (resp.get("result", {}).get("content") or [{}])[0].get("text", "")
    print(f"[{'FAIL' if err else 'OK'}] {name} ({dt:.0f}s): {text.splitlines()[0] if text else ''}",
          flush=True)

proc.stdin.close()
proc.terminate()
print("LOGO GEN DONE", flush=True)
