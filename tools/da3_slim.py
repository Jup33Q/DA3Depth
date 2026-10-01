"""M1: Slim export-friendly wrapper for DA3MONO-LARGE mono depth.

DA3MonoSlim wraps only the two modules used for single-view depth:
  backbone (DinoV2 ViT-L) + head (DPT, output_dim=1 depth + sky head).
Input:  (1, 3, H, W) fp32 tensor, ImageNet-normalized, H/W multiples of 14.
Output: depth_raw (1, H, W), sky (1, H, W) — both BEFORE sky post-processing.

Sky post-processing (set sky pixels to non-sky 0.99-quantile depth) is kept
outside the module (apply_sky_estimation) because it is data-dependent and
non-exportable; it runs in the app / eval harness instead.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from depth_anything_3.utils.alignment import compute_sky_mask, set_sky_regions_to_max_depth


class DA3MonoSlim(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.backbone = net.backbone
        self.head = net.head

    def forward(self, x: torch.Tensor):
        # x: (1, 3, H, W) -> backbone expects (B, N, 3, H, W)
        x5 = x.unsqueeze(1)
        feats, _ = self.backbone(
            x5, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced"
        )
        H, W = x.shape[-2], x.shape[-1]
        B, S, N, C = feats[0][0].shape
        feats_flat = [f[0].reshape(B * S, N, C) for f in feats]
        out = self.head._forward_impl(feats_flat, H, W, patch_start_idx=0)
        return out["depth"], out["sky"]  # each (1, H, W)


def apply_sky_estimation(depth: torch.Tensor, sky: torch.Tensor) -> torch.Tensor:
    """Deterministic replica of DepthAnything3Net._process_mono_sky_estimation
    (full-tensor quantile instead of random sampling)."""
    non_sky_mask = compute_sky_mask(sky, threshold=0.3)
    if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
        return depth
    non_sky_depth = depth[non_sky_mask]
    non_sky_max = torch.quantile(non_sky_depth.float(), 0.99)
    depth_out, _ = set_sky_regions_to_max_depth(depth, None, non_sky_mask, max_depth=non_sky_max)
    return depth_out


def freeze_pos_encoding(slim: DA3MonoSlim, H: int = 504, W: int = 378):
    """Precompute DinoV2 bicubic-interpolated pos_embed for a fixed input size
    and freeze it as a buffer. Bit-identical output, removes upsample_bicubic2d
    (unsupported by coremltools) from the exported graph."""
    vt = slim.backbone.pretrained
    x_dummy = torch.zeros(1, 1 + (H // 14) * (W // 14), vt.embed_dim)
    with torch.no_grad():
        # call-site convention: interpolate_pos_encoding(x, w, h) where w:=H, h:=W
        pe = vt.interpolate_pos_encoding(x_dummy, H, W)  # (1, 1+N, C)
    vt.register_buffer("_frozen_pos_embed", pe.detach().clone(), persistent=False)

    def _frozen(x, w, h):
        return vt._frozen_pos_embed

    vt.interpolate_pos_encoding = _frozen
    return slim


def load_slim_from_hub(model_id: str = "depth-anything/DA3MONO-LARGE"):
    """Load DA3 via official API (HF cache), return (slim_wrapper, full_api_model)."""
    from depth_anything_3.api import DepthAnything3

    api_model = DepthAnything3.from_pretrained(model_id)
    slim = DA3MonoSlim(api_model.model).eval()
    return slim, api_model
