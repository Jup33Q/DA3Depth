"""M6b: CoreAI per-compute-unit benchmark (default / neural_engine / gpu)."""
import asyncio, os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import load_slim_from_hub, freeze_pos_encoding

H, W = 504, 378
IMG = "/Users/jup33q/Desktop/_-__2011559649.png"
AIMODEL = os.path.expanduser(f"~/dev/DA3Depth/models/DA3MonoLarge_{H}x{W}.aimodel")
ITERS = 20

slim, api = load_slim_from_hub()
slim = slim.float().eval()
freeze_pos_encoding(slim, H, W)
imgs, _, _ = api._preprocess_inputs([Image.open(IMG).convert("RGB")], None, None, 504, "upper_bound_resize")
x_np = imgs.float().numpy()

from coreai.runtime import AIModel, NDArray, SpecializationOptions, ComputeUnitKind

variants = {
    "default": None,
    "neural_engine": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine()),
    "gpu": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()),
    "cpu": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.cpu()),
}

for name, opts in variants.items():
    try:
        t0 = time.time()
        ai = asyncio.run(AIModel.load(AIMODEL, specialization_options=opts)) if opts else asyncio.run(AIModel.load(AIMODEL))
        load_s = time.time() - t0
        fn = ai.load_function(ai.function_names[0])
        infer = lambda: asyncio.run(fn({"image": NDArray(x_np)}))
        t0 = time.time(); infer(); first_ms = (time.time() - t0) * 1000
        for _ in range(3): infer()
        t0 = time.time()
        for _ in range(ITERS): infer()
        steady = (time.time() - t0) / ITERS * 1000
        print(f"{name:15s} load {load_s:.1f}s  first {first_ms:.0f} ms  steady {steady:.1f} ms", flush=True)
    except Exception as e:
        print(f"{name:15s} FAILED: {e}", flush=True)
