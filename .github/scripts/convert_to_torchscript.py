#!/usr/bin/env python3
"""
convert_to_torchscript.py

Convert Depth-Anything-v2 (ViT-B) checkpoint (.pth) to a
TorchScript model (.pt) that can be loaded in Foundry Nuke.

Usage
-----
python .github/scripts/convert_to_torchscript.py \
    --checkpoint checkpoints/depth_anything_v2_vitb.pth \
    --output depth_anything_v2_vitb.pt \
    --device cpu
"""
import argparse
import math
from typing import Optional

import torch

# ────────────────────────────────────────────────────────────────────────────
# 1. Monkey-patch Resize.constrain_to_multiple_of so it’s
#    NumPy-free and TorchScript-safe (fixes .astype issue)
# ────────────────────────────────────────────────────────────────────────────
from prior_depth_anything.depth_anything_v2.util.transform import Resize

def _safe_constrain_to_multiple_of(
    self,
    x,
    min_val: int = 0,
    max_val: Optional[int] = None,
) -> int:
    """
    Original implementation used NumPy and .astype(), which breaks tracing.
    This pure-Python version works with Python scalars, NumPy scalars,
    torch.SymInt, or 0-D tensors and always returns an `int`.
    """
    # Convert torch tensors / SymInt to float
    if isinstance(x, torch.Tensor):
        x = float(x.item())
    elif not isinstance(x, (int, float)):
        x = float(x)

    y = round(x / self.__multiple_of) * self.__multiple_of

    if max_val is not None and y > max_val:
        y = math.floor(x / self.__multiple_of) * self.__multiple_of
    if y < min_val:
        y = math.ceil(x / self.__multiple_of) * self.__multiple_of

    return int(y)

# Apply the patch once, globally.
Resize.constrain_to_multiple_of = _safe_constrain_to_multiple_of


# ────────────────────────────────────────────────────────────────────────────
# 2. Build model & TorchScript wrapper
# ────────────────────────────────────────────────────────────────────────────
from prior_depth_anything.depth_anything_v2 import build_backbone

class DepthAnythingWrapper(torch.nn.Module):
    """Fixed, minimal interface for TorchScript export."""

    def __init__(self, core: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.core = core.to(device)
        self.device = device

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        image : uint8 tensor, shape [B, 3, H, W], values 0-255

        Returns
        -------
        depth : float32 tensor, shape [B, 1, H, W], metres
        """
        return self.core(
            image.to(self.device),
            input_size=518,      # canonical size
            condition=None,
            device=self.device,
        )


# ────────────────────────────────────────────────────────────────────────────
# 3. CLI helpers
# ────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Depth-Anything-v2 ViT-B .pth to TorchScript .pt"
    )
    p.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to depth_anything_v2_vitb.pth"
    )
    p.add_argument(
        "--output", default="depth_anything_v2_vitb.pt", type=str,
        help="Destination filename for scripted model"
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to load weights & trace on"
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 4. Entry-point
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # Build backbone & load weights
    model = build_backbone(
        depth_size="vitb",
        encoder_cond_dim=-1,
        model_path=args.checkpoint,
    )
    model.eval()

    wrapper = DepthAnythingWrapper(model, device=args.device)
    wrapper.eval()

    # Dummy RGB tensor for tracing (strict=False keeps H,W symbolic)
    example = torch.zeros(1, 3, 518, 518, dtype=torch.uint8, device=args.device)

    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, example, strict=False)
        scripted.save(args.output)

    print(f"✓ TorchScript model saved → {args.output}")


if __name__ == "__main__":
    main()
