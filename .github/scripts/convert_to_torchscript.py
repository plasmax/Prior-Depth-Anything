#!/usr/bin/env python3
"""
convert_to_torchscript.py

Convert Depth-Anything-v2 (ViT-B) checkpoint (.pth)
to a TorchScript model (.pt) ready for use in
Foundry Nuke’s PyTorch-based nodes.

Example
-------
python convert_to_torchscript.py \
    --checkpoint depth_anything_v2_vitb.pth \
    --output depth_anything_v2_vitb.pt \
    --device cpu
"""

import argparse
import torch

# Import the model builder from the Prior-Depth-Anything repo.
from prior_depth_anything.depth_anything_v2 import build_backbone


class DepthAnythingWrapper(torch.nn.Module):
    """A thin wrapper to give TorchScript a fixed signature."""

    def __init__(self, core: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.core = core.to(device)
        self.device = device

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : uint8 tensor of shape [B, 3, H, W] with values 0‒255

        Returns
        -------
        depth : float32 tensor of shape [B, 1, H, W] in metres
        """
        return self.core(
            image.to(self.device),
            input_size=518,      # network’s canonical size
            condition=None,      # no extra conditioning
            device=self.device,  # keeps internal ops on the same device
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Depth-Anything-v2 ViT-B .pth to TorchScript .pt."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to depth_anything_v2_vitb.pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="depth_anything_v2_vitb.pt",
        help="Destination filename for the scripted model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to load weights & trace on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build the backbone and load weights.
    model = build_backbone(
        depth_size="vitb",
        encoder_cond_dim=-1,
        model_path=args.checkpoint,
    )
    model.eval()  # important for deterministic tracing

    # Wrap for TorchScript.
    wrapper = DepthAnythingWrapper(model, device=args.device)
    wrapper.eval()

    # Dummy inference tensor (518 × 518 RGB).
    example_input = torch.zeros(
        1, 3, 518, 518, dtype=torch.uint8, device=args.device
    )

    # Trace to TorchScript.
    with torch.no_grad():
        scripted = torch.jit.trace(
            wrapper, example_input, strict=False
        )
        scripted.save(args.output)

    print(f"TorchScript model saved to: {args.output}")


if __name__ == "__main__":
    main()
