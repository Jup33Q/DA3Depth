"""M0 smoke test: load DA3MONO-LARGE from local HF cache, run depth inference on one image."""
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3

img_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/m0_smoke"
os.makedirs(out_dir, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device={device} torch={torch.__version__}")

t0 = time.time()
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE").to(device)
print(f"loaded in {time.time()-t0:.1f}s")

img = Image.open(img_path).convert("RGB")
print(f"input: {img_path} size={img.size}")

t0 = time.time()
pred = model.inference([img], process_res=504, process_res_method="upper_bound_resize")
dt = time.time() - t0
depth = pred.depth[0]  # (H, W) numpy, raw relative depth
print(f"inference {dt:.2f}s, depth shape={depth.shape} min={depth.min():.4f} max={depth.max():.4f}")

# near=small -> display: normalize so near=black far=white per spec
d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
gray = Image.fromarray((d * 255).astype(np.uint8), mode="L")
out_path = os.path.join(out_dir, "m0_depth_gray.png")
gray.save(out_path)
np.save(os.path.join(out_dir, "m0_depth_raw.npy"), depth.astype(np.float32))
print(f"saved {out_path}")
print("M0 SMOKE OK")
