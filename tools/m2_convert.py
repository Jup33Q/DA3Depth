"""M2: Export DA3MonoSlim -> CoreML .mlpackage via torch.export + coremltools (FP16, ALL units)."""
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import load_slim_from_hub, freeze_pos_encoding

H = int(sys.argv[1]) if len(sys.argv) > 1 else 504
W = int(sys.argv[2]) if len(sys.argv) > 2 else 378
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(
    f"~/dev/DA3Depth/models/DA3MonoLarge_{H}x{W}.mlpackage"
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("loading model...", flush=True)
slim, _ = load_slim_from_hub()
slim = slim.float().eval()
freeze_pos_encoding(slim, H, W)

example = torch.randn(1, 3, H, W)

print("torch.export...", flush=True)
t0 = time.time()
ep = torch.export.export(slim, (example,))
ep = ep.run_decompositions({})

# strip aten.alias no-op nodes (unsupported by coremltools EXIR frontend)
removed = 0
for n in list(ep.graph.nodes):
    if n.op == "call_function" and n.target == torch.ops.aten.alias.default:
        n.replace_all_uses_with(n.args[0])
        ep.graph.erase_node(n)
        removed += 1
ep.graph.eliminate_dead_code()
ep.graph_module.recompile()
print(f"stripped {removed} alias nodes", flush=True)
print(f"exported in {time.time()-t0:.1f}s", flush=True)

print("coremltools convert...", flush=True)
t0 = time.time()
ml = ct.convert(
    ep,
    inputs=[ct.TensorType(shape=(1, 3, H, W), name="image")],
    outputs=[ct.TensorType(name="depth"), ct.TensorType(name="sky")],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
    convert_to="mlprogram",
)
print(f"converted in {time.time()-t0:.1f}s", flush=True)

ml.author = "DA3Depth"
ml.short_description = "Depth Anything 3 DA3MONO-LARGE mono depth (slim: backbone+DPT head, raw depth+sky)"
ml.input_description["image"] = f"ImageNet-normalized RGB image, (1,3,{H},{W}), fp32"
ml.output_description["depth"] = f"raw relative depth (1,{H},{W}), exp-activated, near=small far=large"
ml.output_description["sky"] = f"sky logits relu (1,{H},{W}); sky if >= 0.3"
ml.save(OUT)
print("saved", OUT)
print("M2 EXPORT OK", flush=True)
