"""M6: CoreAI .aimodel — numeric check vs PyTorch + speed benchmark vs CoreML."""
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import load_slim_from_hub, freeze_pos_encoding, apply_sky_estimation

H = int(sys.argv[1]) if len(sys.argv) > 1 else 504
W = int(sys.argv[2]) if len(sys.argv) > 2 else 378
IMG = sys.argv[3] if len(sys.argv) > 3 else "/Users/jup33q/Desktop/_-__2011559649.png"
AIMODEL = os.path.expanduser(f"~/dev/DA3Depth/models/DA3MonoLarge_{H}x{W}.aimodel")
MLPACKAGE = os.path.expanduser(f"~/dev/DA3Depth/models/DA3MonoLarge_{H}x{W}.mlpackage")
ITERS = int(os.environ.get("ITERS", "20"))


def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64); C1, C2 = 0.01**2, 0.03**2
    ma, mb = uniform_filter(a, 7), uniform_filter(b, 7)
    va = uniform_filter(a*a, 7) - ma**2; vb = uniform_filter(b*b, 7) - mb**2
    cov = uniform_filter(a*b, 7) - ma*mb
    return float((((2*ma*mb+C1)*(2*cov+C2)) / ((ma**2+mb**2+C1)*(va+vb+C2))).mean())


def sky_np(depth, sky):
    ns = sky < 0.3
    if 10 < ns.sum() < ns.size - 10:
        depth = depth.copy()
        depth[~ns] = np.quantile(depth[ns], 0.99)
    return depth


n01 = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)

# ---- PyTorch reference ----
print("preparing pytorch reference...", flush=True)
slim, api = load_slim_from_hub()
slim = slim.float().eval()
freeze_pos_encoding(slim, H, W)
imgs, _, _ = api._preprocess_inputs([Image.open(IMG).convert("RGB")], None, None, 504, "upper_bound_resize")
x = imgs.float()
assert x.shape[-2:] == (H, W), f"{x.shape} != {H}x{W}"
x_np = x.numpy()
with torch.inference_mode():
    d_raw, sky = slim(x)
    d_pt = apply_sky_estimation(d_raw[0].clone(), sky[0]).numpy()

# ---- CoreAI ----
import asyncio
from coreai.runtime import AIModel, NDArray

print("loading .aimodel...", flush=True)
t0 = time.time()
ai = asyncio.run(AIModel.load(AIMODEL))
print(f"load {time.time()-t0:.1f}s, functions: {ai.function_names}", flush=True)
fn = ai.load_function(ai.function_names[0])

def to_ndarray(a):
    return NDArray(np.ascontiguousarray(a))

def ai_infer(xnp):
    out = asyncio.run(fn({"image": to_ndarray(xnp)}))
    d, s = out["depth"].numpy(), out["sky"].numpy()
    return d.reshape(H, W).astype(np.float32), s.reshape(H, W).astype(np.float32)

t0 = time.time()
d_ai, s_ai = ai_infer(x_np)
first_ms = (time.time() - t0) * 1000
d_ai = sky_np(d_ai, s_ai)
print(f"CoreAI vs PyTorch: SSIM={ssim(n01(d_ai), n01(d_pt)):.5f} maxdiff={np.abs(d_ai-d_pt).max():.4f}")

# warmup + bench
for _ in range(3):
    ai_infer(x_np)
t0 = time.time()
for _ in range(ITERS):
    ai_infer(x_np)
ai_ms = (time.time() - t0) / ITERS * 1000

# ---- CoreML ----
import coremltools as ct

ml = ct.models.MLModel(MLPACKAGE, compute_units=ct.ComputeUnit.ALL)
t0 = time.time()
out = ml.predict({"image": x_np})
ml_first_ms = (time.time() - t0) * 1000
d_ml = sky_np(out["depth"][0].astype(np.float32), out["sky"][0].astype(np.float32))
print(f"CoreML vs PyTorch: SSIM={ssim(n01(d_ml), n01(d_pt)):.5f}")
for _ in range(3):
    ml.predict({"image": x_np})
t0 = time.time()
for _ in range(ITERS):
    ml.predict({"image": x_np})
ml_ms = (time.time() - t0) / ITERS * 1000

# ---- PyTorch CPU bench (reference) ----
with torch.inference_mode():
    for _ in range(2):
        slim(x)
    t0 = time.time()
    for _ in range(5):
        slim(x)
    pt_ms = (time.time() - t0) / 5 * 1000

print(f"\n--- benchmark ({H}x{W}, {ITERS} iters) ---")
print(f"CoreAI   : first {first_ms:.0f} ms, steady {ai_ms:.1f} ms")
print(f"CoreML   : first {ml_first_ms:.0f} ms, steady {ml_ms:.1f} ms")
print(f"PyTorch CPU fp32: steady {pt_ms:.1f} ms")
print(f"CoreAI vs CoreML speedup: {ml_ms/ai_ms:.2f}x")
