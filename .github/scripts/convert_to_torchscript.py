#!/usr/bin/env python3
"""
convert_to_torchscript.py

Convert Depth-Anything-v2 (ViT-B) checkpoint (.pth)
to a TorchScript model (.pt) ready for use in
Foundry Nuke’s PyTorch-based nodes.

Usage
-----
python .github/scripts/convert_to_torchscript.py \
    --checkpoint checkpoints/depth_anything_v2_vitb.pth \
    --output depth_anything_v2_vitb.pt \
    --device cpu
"""
import argparse
import math
import torch

# ────────────────────────────────────────────────────────────────────────────
# 1. Hot-patch Resize.constrain_to_multiple_of so that it is TorchScript-safe
#    (avoids NumPy scalar -> .astype() issue during tracing).
# ────────────────────────────────────────────────────────────────────────────
from prior_depth_anything.depth_anything_v2.util.transform import Resize

def _safe_constrain_to_multiple_of(self, x, min_val: int = 0, max_val: int | None = None):
    """Replace original NumPy version with pure-Python arithmetic.

    Accepts Python scalars, NumPy scalars, torch.SymInt, or 0-d tensors.
    Always returns `int`, guaranteeing downstream compatibility.
    """
    if isinstance(x, torch.Tensor):
        # For tracing we will get SymInt / 0-d Tensor → convert to float.
        x = float(x.item())
    elif not isinstance(x, (int, float)):
        x = float(x)

    y = round(x / self.__multiple_of) * self.__multiple_of

    if max_val is not None and y > max_val:
        y = math.floor(x / self.__multiple_of) * self.__multiple_of
    if y < min_val:
        y = math.ceil(x / self.__multiple_of) * self.__multiple_of

    return int(y)

# Apply the monkey-patch once, globally.
Resize.constrain_to_multiple_of = _safe_constrain_to_multiple_of


# ────────────────────────────────────────────────────────────────────────────
# 2. Build model & TorchScript wrapper
# ────────────────────────────────────────────────────────────────────────────
from prior_depth_anything.depth_anything_v2 import build_backbone


class DepthAnythingWrapper(torch.nn.Module):
    """Gives TorchScript a fixed forward signature."""

    def __init__(self, core: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.core = core.to(device)
        self.device = device

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        image : uint8 tensor, shape [B, 3, H, W], values 0-255.

        Returns
        -------
        depth : float32 tensor, shape [B, 1, H, W] in metres.
        """
        return self.core(
            image.to(self.device),
            input_size=518,         # network’s canonical input size
            condition=None,
            device=self.device,
        )


# ────────────────────────────────────────────────────────────────────────────
# 3. CLI helpers
# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
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
    args = parse_args()

    # Build backbone & load weights
    model = build_backbone(
        depth_size="vitb",
        encoder_cond_dim=-1,
        model_path=args.checkpoint,
    )
    model.eval()

    # TorchScript wrapper
    wrapper = DepthAnythingWrapper(model, device=args.device)
    wrapper.eval()

    # Dummy RGB tensor for tracing (strict=False keeps H, W flexible)
    example = torch.zeros(1, 3, 518, 518, dtype=torch.uint8, device=args.device)

    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, example, strict=False)
        scripted.save(args.output)

    print(f"✓ TorchScript model saved → {args.output}")


if __name__ == "__main__":
    main()
