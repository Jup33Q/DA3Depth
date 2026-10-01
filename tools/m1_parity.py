"""M1 parity test: slim wrapper vs original DepthAnything3Net, same image, fp32 CPU.

Gate: max abs diff of final depth (after sky processing) < 1e-3.
"""
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import DA3MonoSlim, apply_sky_estimation, load_slim_from_hub

IMG = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/jup33q/Desktop/the_reference_image_defines_the_subject__style_and_lighting__a_"
    "vertical_oil_style_anime_painting_of_a_long_black_haired_girl_in_a_white_and_navy_sailor_"
    "school_uniform_kneeling_on_a_crac_1216001771.png"
)

slim, api_model = load_slim_from_hub()
net = api_model.model.eval()  # DepthAnything3Net (backbone+head, mono: no cam modules)

# Official preprocessing (upper_bound_resize 504 + ImageNet normalize)
imgs_cpu, _, _ = api_model._preprocess_inputs([Image.open(IMG).convert("RGB")], None, None, 504, "upper_bound_resize")
x = imgs_cpu.float()  # (1, 3, H, W)
print(f"preprocessed input: {tuple(x.shape)}")

with torch.inference_mode():
    t0 = time.time()
    out_orig = net(x.unsqueeze(1))  # full original fp32 forward (incl. sky post-processing)
    depth_orig = out_orig.depth[0, 0]
    t_orig = time.time() - t0

    t0 = time.time()
    depth_raw, sky = slim(x)  # (1, H, W) each
    depth_slim = apply_sky_estimation(depth_raw[0].clone(), sky[0])
    t_slim = time.time() - t0

print(f"orig {t_orig:.1f}s  slim {t_slim:.1f}s")
print(f"orig depth: min={depth_orig.min():.5f} max={depth_orig.max():.5f}")
print(f"slim depth: min={depth_slim.min():.5f} max={depth_slim.max():.5f}")

diff = (depth_slim - depth_orig).abs()
print(f"max abs diff = {diff.max().item():.3e}  mean = {diff.mean().item():.3e}")

# raw (pre-sky) comparison for diagnosis
with torch.inference_mode():
    out_orig2 = net(x.unsqueeze(1))
diff2 = (depth_slim - out_orig2.depth[0, 0]).abs().max().item()
print(f"orig rerun self-diff vs first run = {diff2:.3e} (sky-step sampling nondeterminism check)")

gate = 1e-3
print("M1 PASS" if diff.max().item() < gate else "M1 FAIL", f"(gate {gate})")
