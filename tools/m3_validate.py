"""M3 validation:
1) CoreML (FP16, ALL) vs PyTorch fp32 slim — SSIM > 0.98 gate, per image.
2) PyTorch slim vs ~/da3-output 8-bit reference PNGs — pipeline fidelity check
   (find which input image produced which reference, report SSIM).
"""
import os, sys, glob, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import cv2
import coremltools as ct
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import load_slim_from_hub, freeze_pos_encoding, apply_sky_estimation

MLPACKAGE = os.path.expanduser("~/dev/DA3Depth/models/DA3MonoLarge.mlpackage")
REF_DIR = os.path.expanduser("~/da3-output")
H, W = 504, 378


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Windowed SSIM (7x7 uniform), inputs float in [0,1], same shape."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = uniform_filter(a, 7), uniform_filter(b, 7)
    va = uniform_filter(a * a, 7) - mu_a ** 2
    vb = uniform_filter(b * b, 7) - mu_b ** 2
    cov = uniform_filter(a * b, 7) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def sky_step_np(depth: np.ndarray, sky: np.ndarray) -> np.ndarray:
    non_sky = sky < 0.3
    if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
        return depth
    q = np.quantile(depth[non_sky], 0.99)
    out = depth.copy()
    out[~non_sky] = q
    return out


def norm01(d: np.ndarray) -> np.ndarray:
    return (d - d.min()) / (d.max() - d.min() + 1e-12)


print("loading pytorch slim + coreml model...", flush=True)
slim, api = load_slim_from_hub()
slim = slim.float().eval()
freeze_pos_encoding(slim, H, W)
ml = ct.models.MLModel(MLPACKAGE, compute_units=ct.ComputeUnit.ALL)

# ---- candidate input images ----
candidates = sorted(glob.glob(os.path.expanduser("~/Desktop/*.png")))
refs = sorted(glob.glob(os.path.join(REF_DIR, "*_depth_gray.png")))
print(f"{len(candidates)} candidate inputs, {len(refs)} references")

rows = []
gray_cache = {}
with torch.inference_mode():
    for img_path in candidates:
        name = os.path.basename(img_path)[:40]
        try:
            imgs, _, _ = api._preprocess_inputs(
                [Image.open(img_path).convert("RGB")], None, None, 504, "upper_bound_resize"
            )
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        x = imgs.float()
        if x.shape[-2:] != (H, W):
            print(f"skip {name}: shape {tuple(x.shape)} (not 504x378 portrait)")
            continue
        # PyTorch
        d_raw, sky = slim(x)
        d_pt = apply_sky_estimation(d_raw[0].clone(), sky[0]).numpy()
        # CoreML
        t0 = time.time()
        out = ml.predict({"image": x.numpy()})
        dt = time.time() - t0
        d_ml = sky_step_np(out["depth"][0].astype(np.float32), out["sky"][0].astype(np.float32))
        s = ssim(norm01(d_pt), norm01(d_ml))
        maxd = np.abs(d_pt - d_ml).max()
        rows.append((name, s, maxd, dt))
        gray_cache[img_path] = norm01(d_pt)
        print(f"{name:42s} SSIM={s:.5f} maxabs={maxd:.4f} coreml={dt*1000:.0f}ms", flush=True)

print("\n--- M3 gate: CoreML vs PyTorch SSIM > 0.98 ---")
ok = all(s > 0.98 for _, s, _, _ in rows) and len(rows) > 0
for name, s, maxd, dt in rows:
    print(f"  {'PASS' if s > 0.98 else 'FAIL'} {s:.5f}  {name}")
print("M3 COREML GATE:", "PASS" if ok else "FAIL")

# ---- match references ----
print("\n--- da3-output reference matching (PyTorch gray vs reference gray) ---")
for ref_path in refs:
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    rh, rw = ref.shape
    best = (None, -1)
    for img_path, g in gray_cache.items():
        g_up = cv2.resize(g, (rw, rh), interpolation=cv2.INTER_LINEAR)
        s = ssim(g_up, ref)
        if s > best[1]:
            best = (img_path, s)
    print(f"{os.path.basename(ref_path):45s} best={os.path.basename(best[0])[:40]:42s} SSIM={best[1]:.4f}")
