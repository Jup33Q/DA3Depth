"""M5 end-to-end smoke via DA3Depth built-in MCP server (127.0.0.1:8378).

Flow: initialize -> tools/list -> load_image -> export gray8/color8/gray16
      -> set_flip -> export flipped gray8 -> status
Then verify exported files numerically (dims, bit depth, flip, near-black-far-white,
SSIM vs PyTorch reference at model resolution).
"""
import json, os, sys, urllib.request

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MCP = "http://127.0.0.1:8378/mcp"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jup33q/Desktop/_-__2011559649.png"
OUT = "/tmp/da3_m5"
os.makedirs(OUT, exist_ok=True)


def rpc(method, params=None, _id=[0]):
    _id[0] += 1
    body = json.dumps({"jsonrpc": "2.0", "id": _id[0], "method": method,
                       **({"params": params} if params else {})}).encode()
    req = urllib.request.Request(MCP, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def tool(name, **args):
    r = rpc("tools/call", {"name": name, "arguments": args})
    return r["result"]["content"][0]["text"]


print("initialize:", rpc("initialize")["result"]["serverInfo"])
print("tools:", [t["name"] for t in rpc("tools/list")["result"]["tools"]])

print("load_image:", tool("load_image", path=IMG))
print("set_flip reset:", tool("set_flip", value=False))
print("export gray8 :", tool("export", kind="gray8", path=f"{OUT}/gray.png"))
print("export color8:", tool("export", kind="color8", path=f"{OUT}/color.png"))
print("export gray16:", tool("export", kind="gray16", path=f"{OUT}/depth16.png"))
print("set_flip:", tool("set_flip", value=True))
print("export gray8 flipped:", tool("export", kind="gray8", path=f"{OUT}/gray_flipped.png"))
print("status:", tool("status"))

# ---- numeric verification ----
import numpy as np, cv2, torch
from PIL import Image as PImage
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.expanduser("~/dev/DA3Depth/tools"))
from da3_slim import load_slim_from_hub, freeze_pos_encoding, apply_sky_estimation

def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64); C1, C2 = 0.01**2, 0.03**2
    ma, mb = uniform_filter(a, 7), uniform_filter(b, 7)
    va = uniform_filter(a*a, 7) - ma**2; vb = uniform_filter(b*b, 7) - mb**2
    cov = uniform_filter(a*b, 7) - ma*mb
    return float((((2*ma*mb+C1)*(2*cov+C2)) / ((ma**2+mb**2+C1)*(va+vb+C2))).mean())

g = cv2.imread(f"{OUT}/gray.png", cv2.IMREAD_GRAYSCALE)
f = cv2.imread(f"{OUT}/gray_flipped.png", cv2.IMREAD_GRAYSCALE)
c = cv2.imread(f"{OUT}/color.png")
d16 = cv2.imread(f"{OUT}/depth16.png", cv2.IMREAD_UNCHANGED)
src = cv2.imread(IMG)
ih, iw = src.shape[:2]

checks = []
checks.append(("gray dims == input", g.shape == (ih, iw)))
checks.append(("color dims == input", c.shape == (ih, iw, 3)))
checks.append(("16bit uint16", d16.dtype == np.uint16 and d16.shape == (ih, iw)))
checks.append(("16bit endianness ok", np.abs((d16/257).astype(int) - g.astype(int)).mean() < 2))
fd = np.abs(f.astype(int) - np.fliplr(g).astype(int))
checks.append(("flip correct (maxdiff<=2)", fd.max() <= 2))
h, w = g.shape
center = g[h//3:2*h//3, w//3:2*w//3].mean()
corners = np.concatenate([g[:150, :150].ravel(), g[:150, -150:].ravel(),
                          g[-150:, :150].ravel(), g[-150:, -150:].ravel()]).mean()
checks.append(("near-black far-white", bool(center < corners)))

# SSIM vs PyTorch reference (model resolution, then compare upscaled)
slim, api = load_slim_from_hub()
portrait = ih >= iw
H, W = (504, 378) if portrait else (378, 504)
slim = slim.float().eval(); freeze_pos_encoding(slim, H, W)
imgs, _, _ = api._preprocess_inputs([PImage.open(IMG).convert("RGB")], None, None, 504, "upper_bound_resize")
x = imgs.float()
with torch.inference_mode():
    d_raw, sky = slim(x)
    d_pt = apply_sky_estimation(d_raw[0].clone(), sky[0]).numpy()
n = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
ref8 = (n(d_pt) * 255).astype(np.uint8)
ref_up = cv2.resize(ref8, (iw, ih), interpolation=cv2.INTER_LINEAR)
s = ssim(g.astype(np.float64) / 255, ref_up.astype(np.float64) / 255)
checks.append((f"app gray vs PyTorch SSIM={s:.4f} > 0.98", s > 0.98))

print("\n--- M5 checks ---")
ok = True
for name, passed in checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok &= bool(passed)
print("M5 SMOKE:", "PASS" if ok else "FAIL")
