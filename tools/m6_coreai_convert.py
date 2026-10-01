"""M6: Convert DA3MonoSlim to Core AI (.aimodel) via coreai-torch, validate vs PyTorch,
benchmark vs CoreML. Usage: m6_coreai.py [H W]  (default 504 378 portrait)
"""
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da3_slim import load_slim_from_hub, freeze_pos_encoding

H = int(sys.argv[1]) if len(sys.argv) > 1 else 504
W = int(sys.argv[2]) if len(sys.argv) > 2 else 378
OUT = os.path.expanduser(f"~/dev/DA3Depth/models/DA3MonoLarge_{H}x{W}.aimodel")

from coreai_torch import TorchConverter

print("loading model...", flush=True)
slim, _ = load_slim_from_hub()
slim = slim.float().eval()
freeze_pos_encoding(slim, H, W)

example = torch.randn(1, 3, H, W)
ep = torch.export.export(slim, (example,))
from coreai_torch import get_decomp_table
ep = ep.run_decompositions(get_decomp_table())
removed = 0
for n in list(ep.graph.nodes):
    if n.op == "call_function" and n.target == torch.ops.aten.alias.default:
        n.replace_all_uses_with(n.args[0])
        ep.graph.erase_node(n)
        removed += 1
ep.graph.eliminate_dead_code()
ep.graph_module.recompile()
print(f"exported, stripped {removed} alias nodes", flush=True)

print("coreai-torch convert...", flush=True)
t0 = time.time()
conv = TorchConverter()
conv.add_exported_program(ep, input_names=["image"], output_names=["depth", "sky"])
prog = conv.to_coreai()
print(f"converted in {time.time()-t0:.1f}s", flush=True)

from pathlib import Path
asset = prog.save_asset(Path(OUT))
print("saved", OUT, flush=True)
print("M6 CONVERT OK", flush=True)
