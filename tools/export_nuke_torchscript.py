#!/usr/bin/env python3
"""Export Prior Depth Anything to TorchScript for Nuke CAT files.

This utility wraps :class:`prior_depth_anything.PriorDepthAnything` into a
TorchScript friendly module that consumes five channel images laid out as::

    rgba.red, rgba.green, rgba.blue, rgba.alpha, depth.Z

The wrapper outputs a two channel tensor where channel ``0`` is the refined
depth prediction and channel ``1`` is the validity mask used during inference.

The generated ``.pt`` file can be converted to a ``.cat`` file with Nuke's
``CatFileCreator`` node. Map the input channels to
``rgba.red, rgba.green, rgba.blue, rgba.alpha, depth.Z`` and map the outputs to
``depth.Z`` (depth) and ``rgba.alpha`` (validity mask).
"""

from __future__ import annotations

import argparse
import os

import torch

from prior_depth_anything import PriorDepthAnything


class PriorDepthAnythingCatWrapper(torch.nn.Module):
    """Wrap ``PriorDepthAnything`` with Nuke friendly I/O conventions."""

    def __init__(self, base_model: PriorDepthAnything) -> None:
        super().__init__()
        self.base_model = base_model
        # Minimum positive depth value used by the original implementation.
        self.register_buffer("min_depth", torch.tensor(1.0e-4, dtype=torch.float32))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run the refinement pipeline."""

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4:
            raise RuntimeError("Expected input with shape [N, 5, H, W].")
        if input_tensor.size(1) != 5:
            raise RuntimeError("Expected 5 channels (rgba + depth).")

        output_dtype = input_tensor.dtype
        float_input = input_tensor.to(torch.float32)

        rgb = float_input[:, 0:3, :, :]
        alpha = float_input[:, 3:4, :, :]
        depth_prior = float_input[:, 4:5, :, :]

        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_uint8 = torch.round(rgb * 255.0).to(torch.uint8)

        min_depth = self.min_depth.to(depth_prior.device)
        depth_valid = depth_prior > min_depth
        alpha_valid = alpha > 0.5
        sparse_mask = depth_valid & alpha_valid

        zero_depth = torch.zeros_like(depth_prior)
        sparse_depth = torch.where(sparse_mask, depth_prior, zero_depth)

        pred_depth = self.base_model.forward(
            images=rgb_uint8,
            sparse_depths=sparse_depth,
            sparse_masks=sparse_mask,
            cover_masks=sparse_mask,
            prior_depths=sparse_depth,
            geometric_depths=None,
            pattern=None,
        )

        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)

        refined_depth = pred_depth.to(output_dtype)
        validity = sparse_mask.to(output_dtype)

        return torch.cat((refined_depth, validity), dim=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Prior Depth Anything to TorchScript for Nuke."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the TorchScript module (e.g. priorda_nuke.pt).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for tracing (default: %(default)s).",
    )
    parser.add_argument(
        "--mde-dir",
        default=None,
        help="Optional directory containing frozen backbone weights.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Optional directory containing fine stage checkpoints.",
    )
    parser.add_argument(
        "--frozen-model-size",
        default="vitb",
        choices=["vits", "vitb", "vitl"],
        help="Backbone size for the coarse stage (default: %(default)s).",
    )
    parser.add_argument(
        "--conditioned-model-size",
        default="vitb",
        choices=["vits", "vitb"],
        help="Backbone size for the refinement stage (default: %(default)s).",
    )
    parser.add_argument(
        "--version",
        default="1.1",
        choices=["1.0", "1.1"],
        help="Model version identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="Skip the fine stage and export the coarse model only.",
    )
    parser.add_argument(
        "--example-height",
        type=int,
        default=518,
        help="Height of the dummy tensor used for tracing (default: %(default)s).",
    )
    parser.add_argument(
        "--example-width",
        type=int,
        default=518,
        help="Width of the dummy tensor used for tracing (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-device",
        action="store_true",
        help="Save the TorchScript module on the tracing device instead of CPU.",
    )
    parser.add_argument(
        "--use-script",
        action="store_true",
        help="Compile with torch.jit.script instead of torch.jit.trace.",
    )
    return parser.parse_args()


def _build_example(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic example tensor for tracing/script conversion."""

    example = torch.zeros(1, 5, height, width, dtype=torch.float32, device=device)
    rgb = torch.rand(1, 3, height, width, device=device)
    example[:, 0:3, :, :] = rgb
    example[:, 3:4, :, :] = 1.0
    depth_pattern = torch.linspace(1.0, 2.0, steps=height, device=device).view(1, 1, height, 1)
    example[:, 4:5, :, :] = depth_pattern
    return example


def export_torchscript(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but a CUDA device was requested.")

    priorda = PriorDepthAnything(
        device=args.device,
        version=args.version,
        mde_dir=args.mde_dir,
        ckpt_dir=args.ckpt_dir,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        coarse_only=args.coarse_only,
    )
    priorda.eval()

    wrapper = PriorDepthAnythingCatWrapper(priorda)
    wrapper.eval()

    trace_device = torch.device(args.device)
    wrapper.to(trace_device)

    example = _build_example(args.example_height, args.example_width, trace_device)

    if args.use_script:
        scripted = torch.jit.script(wrapper)
    else:
        with torch.no_grad():
            scripted = torch.jit.trace(wrapper, example, strict=False)

    save_target = trace_device if args.keep_device else torch.device("cpu")
    scripted.to(save_target)

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    scripted.save(output_path)


def main() -> None:
    args = _parse_args()
    export_torchscript(args)


if __name__ == "__main__":
    main()
