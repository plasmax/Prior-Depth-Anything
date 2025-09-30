# prior_depth_anything Python Sources

```
prior_depth_anything
├── depth_anything_v2
│   ├── dinov2_layers
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── drop_path.py
│   │   ├── layer_scale.py
│   │   ├── mlp.py
│   │   ├── patch_embed.py
│   │   └── swiglu_ffn.py
│   ├── util
│   │   ├── blocks.py
│   │   └── transform.py
│   ├── __init__.py
│   ├── dinov2.py
│   └── dpt.py
├── __init__.py
├── cli.py
├── depth_completion.py
├── plugin.py
├── sparse_sampler.py
└── utils.py
```

### prior_depth_anything/__init__.py

```python
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from huggingface_hub import hf_hub_download
from datetime import datetime
from PIL import Image
import glob
from typing import Union, Optional
import time

from .depth_anything_v2 import build_backbone
from .depth_completion import DepthCompletion
from .sparse_sampler import SparseSampler
from .utils import (
    log_img,
    depth2disparity, 
    disparity2depth,
    Arguments
)

class PriorDepthAnything(nn.Module):
    VERSIONS = {
        '1.0': ('', 'error'),
        '1.1': ('_1_1', 'spmask')
    }

    def __init__(self, 
        device: str = 'cuda:0', 
        version: str = '1.1',
        mde_dir: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        frozen_model_size: Optional[str] = None, 
        conditioned_model_size: Optional[str] = None,
        coarse_only: bool = False
    ):
        super(PriorDepthAnything, self).__init__()
        self.args = Arguments()
        postfix, extra_condition = self.VERSIONS[version]
        self.args.extra_condition = extra_condition
        
        self.device = device

        """ 
        For inference stability, we set the output coarse/fine globally. 
        TODO : You can easily modify the code to specify the model to output coarse/fine depth sample-wisely.
        """
        self.coarse_only = coarse_only
        if frozen_model_size:
            self.args.frozen_model_size = frozen_model_size
        if conditioned_model_size:
            self.args.conditioned_model_size = conditioned_model_size
        
        ## Frozon MDE loading.
        if self.args.frozen_model_size in ['vitg']:
            raise ValueError(f'{self.args.frozen_model_size} coming soon...')
        fmde_name = f'depth_anything_v2_{self.args.frozen_model_size}.pth' # Download model checkpoints
        if mde_dir is None:
            fmde_path = hf_hub_download(repo_id=self.args.repo_name, filename=fmde_name)
        else:
            fmde_path = os.path.join(mde_dir, fmde_name)
        print(f"Loading pretrained fmde from {fmde_path}...")
        
        # Initialize Frozon-MDE.
        self.completion = DepthCompletion.build(args=self.args, fmde_path=fmde_path, device=device)
        
        ## Conditioned MDE loading.
        if not coarse_only:
            if self.args.conditioned_model_size in ['vitl', 'vitg']:
                raise ValueError(f'{self.args.conditioned_model_size} coming soon...')
        
            # Initialize and load preptrained `prior-depth-anything` models.
            model = build_backbone(
                depth_size=self.args.conditioned_model_size, 
                encoder_cond_dim=3
            )
            model.construct_aux_layers()

            self.model = self.load_checkpoints(model, ckpt_dir, postfix, self.device).eval()
            
        self.sampler = SparseSampler(device=device, completion=self.completion)
    
    def load_checkpoints(self, model, ckpt_dir, postfix='', device='cuda:0'):
        ckpt_name = f'prior_depth_anything_{self.args.conditioned_model_size}{postfix}.pth'
        if ckpt_dir is None:
            ckpt_path = hf_hub_download(repo_id=self.args.repo_name, filename=ckpt_name)
        else:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"Loading checkpoint from {ckpt_path}...")
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        new_state_dict = OrderedDict()
        for key, value in state_dict['model'].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        return model
        
    def forward(self, 
            images: torch.Tensor, 
            sparse_depths: torch.Tensor, 
            sparse_masks: torch.Tensor, 
            cover_masks: torch.Tensor = None, 
            prior_depths: torch.Tensor = None, 
            geometric_depths: torch.Tensor = None, 
            pattern: Optional[str] = None
        ):
        """ To facilitate further research, we batchify the forward process. """
        ##### Coarse stage. #####
        completed_maps = self.completion(
            images=images, 
            sparse_depths=sparse_depths, 
            sparse_masks=sparse_masks, 
            cover_masks=cover_masks, 
            prior_depths=prior_depths, 
            pattern=pattern,
            geometric_depths=geometric_depths
        )
        
        # knn-aligned depths
        comp_cond = completed_maps['scaled_preds'].unsqueeze(1)
        if self.coarse_only:
            coarse_depths = disparity2depth(comp_cond)
            return coarse_depths
        # Global Scale-Shift aligned depths.
        global_cond = completed_maps['global_preds'].unsqueeze(1)
        
        ##### Fine stage. #####
        if self.args.normalize_depth:
            # Obtain the value of norm params.
            masked_min, denom = self.zero_one_normalize(sparse_depths, sparse_masks, affine_only=True)
            
            global_depths = (disparity2depth(global_cond) - masked_min) / denom
            global_cond = depth2disparity(global_depths)
            
            comp_depths = (disparity2depth(comp_cond) - masked_min) / denom
            comp_cond = depth2disparity(comp_depths)
        condition = torch.cat([global_cond, comp_cond], dim=1)
        
        if self.args.extra_condition == 'error':
            uctns = completed_maps['uncertainties'].unsqueeze(1)
            condition = torch.cat([uctns, condition], dim=1)
        elif self.args.extra_condition == 'spmask':
            condition = torch.cat([sparse_masks, condition], dim=1)
        else:
            raise NotImplementedError(
                "The extra condition can only be `error_map` or `sparse_map`.")
            
        # heit = sparse_depths.shape[-2] // 14 * 14
        heit = 518
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t0 = time.time()
        metric_disparities = self.model(images, heit, condition=condition, device=self.device)
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t1 = time.time()
            self.timer.append(t1 - t0)
            
        metric_depths = disparity2depth(metric_disparities)
        if self.args.normalize_depth:
            metric_depths = metric_depths * denom + masked_min
        return metric_depths
    
    def zero_one_normalize(self, depth_maps, valid_masks=None, affine_only=False):
        
        if valid_masks is not None:
            masked_min = depth_maps.masked_fill(~valid_masks, float('inf')).min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.masked_fill(~valid_masks, float('-inf')).max(dim=-1).values.max(dim=-1).values  # (B, 1)
        else:
            masked_min = depth_maps.min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.max(dim=-1).values.max(dim=-1).values  # (B, 1)
        
        denom = masked_max - masked_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        masked_min = masked_min.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        denom = denom.view(-1, 1, 1, 1)
        
        if not affine_only:
            normalized = (depth_maps - masked_min) / denom
            return normalized, (masked_min, denom)
        else:
            return masked_min, denom
    
    def analyze_results(self, prior_depth, pred_depth, sparse_depth, log_dir, dir_name):
        """
        We visualize depth prior here, (gt_depth or prior_depth may be stored in uint16). 
            1. If there is ground-truth depth, we visualize ground-truth. 
            2. If the provided depth map is sampled depth prior, 
                we visualize the prior depth map.
        """
        
        print("Saving visual results...")
        prior_depth = prior_depth.squeeze().cpu().numpy()
        scale, shift = prior_depth.max() - prior_depth.min(), prior_depth.min()
        
        gt_name = os.path.join(dir_name, 'gt_depth.*')
        gt_path = glob.glob(gt_name, recursive=False)
        if gt_path:
            gt_path = gt_path[0]
            if os.path.exists(gt_path):
                if gt_path.split('.')[-1] in ['png', 'jpg']:
                    gt_depth = np.asarray(Image.open(gt_path)).astype(np.float32)
                elif gt_path.endswith('npy'):
                    gt_depth = np.load(gt_path)
                else:
                    raise NotImplementedError
                
                scale, shift = gt_depth.max() - gt_depth.min(), gt_depth.min()
                
                log_img(
                    gt_depth.squeeze(),
                    os.path.join(log_dir, 'gt_norm.png'),
                    valids = gt_depth > 0.0001,
                    scale=scale, shift=shift
                )
        
        log_img(
            prior_depth,
            os.path.join(log_dir, 'prior_norm.png'),
            valids=prior_depth > 0.0001,
            scale=scale, shift=shift
        )
        
        log_img(
            pred_depth.squeeze().cpu().numpy(),
            os.path.join(log_dir, 'pred_depth.png'), 
            scale=scale, shift=shift
        )
        
        sparse_depth = sparse_depth.squeeze().cpu().numpy()
        log_img(
            sparse_depth,
            os.path.join(log_dir, 'sparse_depth.png'),
            valids = sparse_depth > 0.0001,
            scale=scale, shift=shift
        )
        
    @torch.no_grad()
    def infer_one_sample(self, 
        image: Union[str, torch.Tensor, np.ndarray] = None, 
        prior: Union[str, torch.Tensor, np.ndarray] = None, 
        geometric: Union[str, torch.Tensor, np.ndarray, None] = None,
        pattern: str = None, 
        double_global: bool = False, 
        prior_cover: bool = False, 
        visualize: bool = False,
        down_fill_mode: str = 'linear'
    ) -> torch.Tensor:
        """ Perform inference. Return the refined/completed depth.
        
        Args:
            image: 
                1. RGB in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Image path of RGB
            prior:
                1. Prior depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of prior depth map. (with scale)
            geometric:
                1. Geometric depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of geometric depth map. (with geometry)
            pattern: The mode of prior-based additional sampling. It could be None.
            double_global: Whether to condition with two estimated depths or estimated + knn-map.
            prior_cover: Whether to keep all prior areas in knn-map, it functions when 'pattern' is not None.
            visualize: Save results. 
            
            
            Example1:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-2/rgb.jpg'
                >>> prior_path = 'assets/sample-2/prior_depth.png'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
                
            Example2:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-6/rgb.npy'
                >>> prior_path = 'assets/sample-6/prior_depth.npy'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
        """
        
        # For each inference, params below should be reset.
        self.args.double_global = double_global
        assert image is not None and prior is not None
        
        ### Load and preprocess example images
        # We implement preprocess with batch size of 1, but our model works for multi-images naturally.
        data = self.sampler(
            image=image, 
            prior=prior,
            geometric=geometric,
            pattern=pattern, 
            K=self.args.K,
            prior_cover=prior_cover,
            down_fill_mode=down_fill_mode
        )
        rgb, prior_depth, sparse_depth = data['rgb'], data['prior_depth'], data['sparse_depth'] # Shape: [B, C, H, W]
        cover_mask, sparse_mask = data['cover_mask'], data['sparse_mask'] # Shape: [B, 1, H, W]
        geometric_depth = data['geometric_depth'] if geometric is not None else None
        if (sparse_mask.view(sparse_mask.shape[0], -1).sum(dim=1) < self.args.K).any():
            raise ValueError("There are not enough known points in at least one of samples")

        ### The core inference stage.
        """ If you want to input multiple samples at once, just stack samples at dim=0, s.t. [B, C, H, W] """
        pred_depth = self.forward(
            images=rgb, 
            sparse_depths=sparse_depth, 
            prior_depths=prior_depth,
            sparse_masks=sparse_mask, 
            cover_masks=cover_mask, 
            pattern=pattern,
            geometric_depths=geometric_depth
        ) # (B, 1, H, W)
        
        ### Visualize the results.
        if visualize:
            # 'dir_name' is the path that stores Ground-Truth RGB and depth.
            if isinstance(image, str):
                dir_name = os.path.dirname(image)
            else:
                dir_name = '.'
                
            parent = datetime.now().strftime("%Y-%m-%d %H:%M")
            log_dir = os.path.join(self.args.log_dir, parent)
            os.makedirs(log_dir, exist_ok=True)
            self.analyze_results(prior_depth, pred_depth, sparse_depth, log_dir, dir_name)
        
        return pred_depth.squeeze()
```

### prior_depth_anything/cli.py

```python
import torch
import argparse
from . import PriorDepthAnything

def create_and_execute():
    parser = argparse.ArgumentParser(
        prog="priorda",
        description="Setting."
    )
    
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True,
        help="Now, only inference is available."
    )
    
    test_parser = subparsers.add_parser(
        "test",
        help="Run inference"
    )
    ## Model settings.
    test_parser.add_argument(
        "--coarse_only", 
        default=0, 
        type=bool,
        help="If specified True, predict without the fine stage.")
    test_parser.add_argument(
        "--frozen_model_size", 
        default='vitb', 
        type=str,
        help="Size of model in coarse stage.")
    test_parser.add_argument(
        "--conditioned_model_size", 
        default='vitb', 
        type=str,
        help="Size of model in fine stage.")
    
    ## Case settings.
    test_parser.add_argument(
        "--image_path", 
        required=True, 
        type=str,
        help="Path of RGB. e.g. assets/sample-1/rgb.jpg")
    
    test_parser.add_argument(
        "--prior_path", 
        required=True, 
        type=str,
        help="Path of Prior depth. e.g. assets/sample-1/gt_depth.png")
    
    test_parser.add_argument(
        "--geometric_path", 
        default=None, 
        type=str,
        help="(Optional) Path of geometric depth. e.g. asserts/sample-1/geo_depth.npy")
    
    test_parser.add_argument(
        "--pattern", 
        default=None, 
        type=str,
        help="(Optional) Pattern for sampling sparse depth points additionally in `prior`. If None, the prior depth is used.")
    
    test_parser.add_argument(
        "--visualize", 
        type=int, 
        default=1, 
        help="Whether to visualize the results.")
    
    test_parser.add_argument(
        "--down_fill_mode",
        type=str,
        default='linear',
        help=(
            "The mode to fill in the vacancy in the prior. Only works for `pattern='^downscale_\d*$'`. "
            "Choices=('knn', 'global', 'linear')"
        )
    )
    test_parser.set_defaults(func=test)
    
    args = parser.parse_args()
    args.func(args)
    
    
def test(args):
    """ To test with models of different sizes, please specify `frozen_model_size` and `conditioned_model_size` """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    priorda = PriorDepthAnything(
        device=device, 
        coarse_only=args.coarse_only,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size
    ) 
    
    """
    image: 
        The path of the image (e.g., '*.jpg') or a tensor/array representing the image. 
        Shape should be [H, W, 3] with values in the range [0, 255].

    prior: 
        The path of the prior depth (e.g., '*.png') or a tensor/array representing the prior depth.
        Shape should be [H, W] with type float32.
        
    geometric (optional): 
        The path of the geometric depth (e.g., '*.png') or a tensor/array representing the geometric depth.
        Shape should be [H, W] with type float32.

    pattern (optional): 
        Pattern for sampling sparse depth points additionally in `prior`. If None, the prior depth is used.
    """
    output = priorda.infer_one_sample(
        image=args.image_path, 
        prior=args.prior_path, 
        geometric=args.geometric_path,
        pattern=args.pattern, 
        visualize=args.visualize,
        down_fill_mode=args.down_fill_mode
    )
    
    
```

### prior_depth_anything/depth_anything_v2/__init__.py

```python
from .dpt import DepthAnythingV2
import torch
import os

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def build_backbone(depth_size='vitb', encoder_cond_dim=-1):
    return DepthAnythingV2(**model_configs[depth_size], encoder_cond_dim=encoder_cond_dim)
```

### prior_depth_anything/depth_anything_v2/dinov2.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None, condition=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x, condition=condition)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1, condition=None):
        x = self.prepare_tokens_with_masks(x, condition=condition)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
                
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        condition: torch.Tensor = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n, condition=condition)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOv2(model_name):
    model_zoo = {
        "vits": vit_small, 
        "vitb": vit_base, 
        "vitl": vit_large, 
        "vitg": vit_giant2
    }
    
    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1
    )
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/__init__.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .block import NestedTensorBlock
from .attention import MemEffAttention
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/attention.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/block.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/drop_path.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py


from torch import nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/layer_scale.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/mlp.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/patch_embed.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def init_alpha_conv(self, cond_channels=2):
        self.alpha_proj = nn.Conv2d(cond_channels, self.embed_dim, 
                kernel_size=self.patch_size, stride=self.patch_size)
        nn.init.constant_(self.alpha_proj.weight, 0)
        nn.init.constant_(self.alpha_proj.bias, 0)

    def forward(self, x: Tensor, condition=None) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        if hasattr(self, 'alpha_proj'):
            x += self.alpha_proj(condition)
        
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
```

### prior_depth_anything/depth_anything_v2/dinov2_layers/swiglu_ffn.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
```

### prior_depth_anything/depth_anything_v2/dpt.py

```python
import pdb

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import numpy as np

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import NormalizeImage, Resize

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        encoder_cond_dim=-1
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.encoder_cond_dim = encoder_cond_dim
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features 
        head_features_2 = 32
        
        hido_feature = head_features_1 // 2
        hidi_feature = hido_feature
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, hido_feature, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(hidi_feature, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w, condition=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out

class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        encoder_cond_dim=-1
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.encoder_cond_dim = encoder_cond_dim
        self.pretrained = DINOv2(model_name=encoder)
        self.out_channels = features // 2
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, 
                                  use_clstoken=use_clstoken, encoder_cond_dim=encoder_cond_dim)

    def forward(self, image, input_size=518, condition=None, device='cuda:0'):
        x, (h, w) = self.raw2input(image, input_size, device)
        
        rh, rw = x.shape[-2:]
        patch_h, patch_w = rh // 14, rw // 14
        
        if self.encoder_cond_dim > 0: 
            condition = F.interpolate(condition, (rh, rw), mode="bilinear", align_corners=True)
        else: 
            condition = None
            
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True, condition=condition)
        disparity = self.depth_head(features, patch_h, patch_w, condition=condition)
        
        disparity = F.relu(disparity).squeeze(1)
        disparity = F.interpolate(disparity[:, None], (h, w), mode="bilinear", align_corners=True)     
        
        return disparity
    
    def freeze_network(self, names: dict):
        trainable_params = {
            'encoder': self.pretrained,
            'decoder': self.depth_head
        }
        
        for name in names:
            if name in trainable_params:
                print(f'Freezing the {name} now.')
                for param in trainable_params[name].parameters():
                    param.requires_grad = False
            else:
                print('Please input an existing parameters\' name...' )
                
    def construct_aux_layers(self):
        self.depth_head.scratch.output_conv2 = nn.Sequential(
            self.depth_head.scratch.output_conv2,
            nn.ReLU(),
            nn.Identity()
        )
        if self.encoder_cond_dim > 0:
            self.pretrained.patch_embed.init_alpha_conv(cond_channels=self.encoder_cond_dim)
        
        if hasattr(self.depth_head.scratch.refinenet4, 'resConfUnit1'):
            del self.depth_head.scratch.refinenet4.resConfUnit1
        if hasattr(self.pretrained, 'mask_token'):
            del self.pretrained.mask_token
    
    def raw2input(self, raw_image, input_size=518, device='cuda'):
        assert isinstance(raw_image, torch.Tensor)
        assert raw_image.dtype == torch.uint8
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method='bicubic',
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                device=device
            )
        ])
        raw_image = raw_image.to(device)
        
        h, w = raw_image.shape[-2:]
        raw_image = raw_image[:, [2, 1, 0], :, :] / 255.0 
        images = transform({'image': raw_image})['image']
        return images, (h, w)
```

### prior_depth_anything/depth_anything_v2/util/blocks.py

```python
import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output
```

### prior_depth_anything/depth_anything_v2/util/transform.py

```python
import torch
import cv2
import numpy as np

class Resize(object):
    def __init__(
        self, 
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method='bilinear',
    ):
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        
    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width
        
        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise NotImplementedError()
            
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        else:
            raise NotImplementedError()
        
        return (new_width, new_height)
    
    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        return y

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[-1], sample["image"].shape[-2])
        sample["image"] = torch.nn.functional.interpolate(
            sample["image"], (height, width), mode=self.__image_interpolation_method)
        
        return sample
    
class NormalizeImage(object):
    def __init__(self, mean, std, device='cpu'):
        self.__mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample
```

### prior_depth_anything/depth_completion.py

```python
import torch
import re
import torch_cluster
import warnings
import time
from typing import Dict, Tuple, Optional

from .utils import (
    depth2disparity,
    disparity2depth
)

class DepthCompletion(torch.nn.Module):
    @staticmethod
    def build(**kwargs):
        return DepthCompletion(**kwargs)
    
    def __init__(self, args, fmde_path, device=None):
        super().__init__()
        
        self.args = args
        self.K = args.K
        
        self.set_device(device)
        self.depth_model = self.init_depth_model(fmde_path)
        
    def set_device(self, device=None):
        if device is not None:
            self.device = device
            return
        
        if torch.cuda.is_available(): 
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
    def unify_format(self, 
        images: torch.Tensor, 
        sparse_depths: torch.Tensor, 
        sparse_masks: torch.Tensor, 
        cover_masks: Optional[torch.Tensor], 
        prior_depths: Optional[torch.Tensor], 
        geometric_depths: Optional[torch.Tensor]
    ):
        # Tune the shape of the tensors.
        if images.max() <= 1: images = (images * 255)
        if images.dtype != torch.uint8: images = images.to(torch.uint8)
        if len(sparse_depths.shape) == 4: 
            sparse_depths = sparse_depths.squeeze(dim=1)
        if len(sparse_masks.shape) == 4:
            sparse_masks = sparse_masks.squeeze(dim=1)
        
        if cover_masks is not None and len(cover_masks.shape) == 4:
            cover_masks = cover_masks.squeeze(dim=1)
        if prior_depths is not None and len(prior_depths.shape) == 4:
            prior_depths = prior_depths.squeeze(dim=1)
        if geometric_depths is not None and len(geometric_depths.shape) == 4:
            geometric_depths = geometric_depths.squeeze(dim=1)
            
        # Move the tensors to the target device.
        images = images.to(self.device)
        sparse_depths, sparse_masks = sparse_depths.to(self.device), sparse_masks.to(self.device)
        
        return images, sparse_depths, sparse_masks, cover_masks, prior_depths, geometric_depths
    
    @torch.no_grad()
    def preprocess(self, 
        images: torch.Tensor, 
        sparse_depths: torch.Tensor, 
        sparse_masks: torch.Tensor, 
        cover_masks: Optional[torch.Tensor] = None, 
        prior_depths: Optional[torch.Tensor] = None, 
        geometric_depths: Optional[torch.Tensor] = None
    ):
        """
        1. Unify the format of all the inputs.
        2. Obtain the model-predicted affine-invariant depth map.
        3. Convert the ground-truth depth to disparity.
        """
        int_images, sparse_depths, sparse_masks, cover_masks, prior_depths, geometric_depths = self.unify_format(
            images, sparse_depths, sparse_masks, cover_masks, prior_depths, geometric_depths)
        
        # Preprocess pred_disparities.
        if geometric_depths is not None:
            warnings.warn("The geometric depth is provided by the user. ")
            pred_disparities = depth2disparity(geometric_depths)
        else:
            # heit = sparse_depths.shape[-2] // 14 * 14
            heit = 518
            if hasattr(self, "timer"):
                torch.cuda.synchronize()
                t0 = time.time()
            pred_disparities = self.depth_model(int_images, heit, device=self.device)
            if hasattr(self, "timer"):
                torch.cuda.synchronize()
                t1 = time.time()
                self.timer.append(t1 - t0)
                
            pred_disparities = pred_disparities.squeeze(1)
            
        # Preprocess sparse_depths and prior depths.
        sparse_disparities = depth2disparity(sparse_depths)
        if prior_depths is not None:
            prior_disparities = depth2disparity(prior_depths)
        else:
            prior_disparities = None
        
        return pred_disparities, sparse_disparities, sparse_masks, cover_masks, prior_disparities
        
    @torch.no_grad()
    def forward(self, 
        images: torch.Tensor, 
        sparse_depths: torch.Tensor, 
        sparse_masks: torch.Tensor, 
        cover_masks: Optional[torch.Tensor] = None, 
        prior_depths: Optional[torch.Tensor] = None, 
        geometric_depths: Optional[torch.Tensor] = None, 
        pattern: Optional[str] = None, 
        ret: str = 'all' # ret = 'knn' or 'global'
    ) -> Dict[str, torch.Tensor]:
        """
        Processe input images and sparse depth information to produce completed depth maps.
        We use global alignment and KNN alignment to refine the depth predictions.
    
        Args:
            images (torch.Tensor): The input images.
            sparse_depths (torch.Tensor): The sparse depth information.
            sparse_masks (torch.Tensor): Indicating which points in the sparse depth are valid.
            cover_masks (torch.Tensor, optional): Indicating areas to be covered by prior depth.
            prior_depths (torch.Tensor, optional): Prior depth information for covering large areas.
            pattern (optional): Pattern for sampling sparse depth points.
    
        Returns:
            Dict[str, torch.Tensor]: Containing the processed data, including:
                - 'uncertainties': A tensor representing the uncertainty of the depth predictions.
                - 'scaled_preds': A tensor representing the scaled depth predictions.
                - 'global_preds': A tensor representing the globally aligned depth predictions.
        """
        assert ret in ['global', 'knn', 'all'], "Unknown return type."
        
        pred_disparities, sparse_disparities, sparse_masks, cover_masks, prior_disparities = self.preprocess(
            images, sparse_depths, sparse_masks, cover_masks, prior_depths, geometric_depths
        )
        
        output = {}
        
        # The masks denote the areas to be completed. Exclude the sparse points to accelerate.
        complete_masks = torch.ones_like(sparse_masks).to(torch.bool)
        complete_masks[sparse_masks] = False
        
        
        # ================================== Global Alignment.
        if ret != 'knn':
            global_preds = self.ss_completer(
                sparse_disparities=sparse_disparities,
                pred_disparities=pred_disparities,
                sparse_masks=sparse_masks
            )
            if ret == 'global':
                # 
                global_preds[sparse_masks] = sparse_disparities[sparse_masks]
                return global_preds
            output['global_preds'] = global_preds
        
        
        # ================================== KNN Alignments.
        if self.args.double_global:
            assert ret == 'all'
            scaled_preds = global_preds.clone()
            scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
        else:
            # Scale the pred_disparities with KNN alignment.
            scaled_preds = self.kss_completer(
                sparse_disparities=sparse_disparities,
                pred_disparities=pred_disparities,
                sparse_masks=sparse_masks, 
                complete_masks=complete_masks,
                K=self.K,
            )
            
        """ 
        Notes: The sparse points have been covered in the kss-completer. 
        And we keep the 
        """
        if cover_masks is not None and cover_masks.sum() > 0:
            # To cover the large areas that have been known.
            scaled_preds[cover_masks] = prior_disparities[cover_masks]
        elif not pattern:
            warnings.warn(
                "The depth prior is directly provided by the user. All the known points will cover the knn-scaled map.")
        else:
            assert not re.fullmatch(r'^cubic_\d+$', pattern)
            assert not re.fullmatch(r'^distance_\d+_\d+$', pattern)
        if ret == 'knn':
            return scaled_preds
        output['scaled_preds'] = scaled_preds
        
        
        # ================================== Process the Uncertainty map.
        if self.args.extra_condition == 'error':
            cal_mask = (global_preds > 0.)
            masked_scaled, scaled_global = scaled_preds[cal_mask], global_preds[cal_mask]
            uctn = torch.abs(masked_scaled - scaled_global) / scaled_global
            uncertainties = torch.zeros_like(scaled_preds, dtype=torch.float32)
            uncertainties[cal_mask] = uctn

            # If needed, normalize the Uncertainty.
            if self.args.normalize_confidence:
                uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
            output['uncertainties'] = uncertainties
        
        return output
        
    def init_depth_model(self, fmde_path):
        """ We implement @depth-anything-v2 here, you can replace it with other depth estimation models. (like VGGT or moge ...)"""
        from .depth_anything_v2 import build_backbone
        depth_model = build_backbone(
            depth_size=self.args.frozen_model_size
        )
        state_dict = torch.load(fmde_path, map_location='cpu')
        depth_model.load_state_dict(state_dict=state_dict)
        
        depth_model.construct_aux_layers()
        depth_model.freeze_network({'encoder', 'decoder'})
        depth_model = depth_model.eval().to(self.device)
        
        return depth_model

        """
        ### For VGGT: (Please also modify the call.)
        from vggt.models.vggt import VGGT
        from .depth_anything_v2.util.transform import Resize

        # Initialize vggt.
        print("Initialize VGGT and load the pretrained weights.")
        vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        vggt = vggt.to(self.device).eval()
        
        @torch.no_grad()
        def depth_model(images, input_size=518):
            images = images.to(torch.float32) / 255.0
            oh, ow = images.shape[-2:]
            images = Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method='bicubic',
            )({'image': images})['image']

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = vggt.aggregator(images)
            # Predict Depth Maps
            depth_map, depth_conf = vggt.depth_head(aggregated_tokens_list, images, ps_idx)
            import torch.nn.functional as F
            depths = F.interpolate(
                depth_map.squeeze(-1), size=(oh, ow), mode='bilinear', align_corners=True
            ).squeeze(1)
            disparities = depth2disparity(depths)
            
            return disparities
        print("VGGT loaded!")
        return depth_model
        
        """
    
    def calc_scale_shift(self, 
        k_sparse_targets: torch.Tensor, 
        k_pred_targets: torch.Tensor, 
        currk_dists: Optional[torch.Tensor] = None, 
        knn: bool = False
    ):
        k_pred_targets += torch.rand(*k_pred_targets.shape, device=self.device) * 1e-5
        X = torch.stack([k_pred_targets, torch.ones_like(k_pred_targets, device=self.device)], dim=2)
        
        # To perform weights to the knn points.
        if knn > 0: k_sparse_targets, X = self.perform_weighted(k_sparse_targets, X, currk_dists)
        elif k_pred_targets.shape[0] > 1: k_sparse_targets = k_sparse_targets.unsqueeze(-1)
        
        solution = torch.linalg.lstsq(X, k_sparse_targets)
        scale, shift = solution[0][:, 0].squeeze(), solution[0][:, 1].squeeze()
        
        return scale, shift
    
    def perform_weighted(self, 
        sparse_ori : torch.Tensor, 
        pred_ori : torch.Tensor, 
        dists : torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform weighted operations on input tensors using distance-based weights. A diagonal 
        matrix is created from the normalized weights and used to weight the inputs.
        
        Notes:
            - Weights are calculated as the inverse of the distances.
            - Weights are normalized to ensure they sum to 1.
 
        Args:
            sparse_ori (torch.Tensor): Sparse original map.
            pred_ori (torch.Tensor): Predicted map.
            dists (torch.Tensor): Distances used for weight calculation.
 
        Returns:
            Tuple: Containing two tensors:
                - sparse_weighted: The weighted version of the sparse original map.
                - pred_weighted: The weighted version of the predicted map.
        """
        
        weights = 1 / dists
        wsum = weights.sum(dim=1, keepdim=True)
        weights = weights / wsum
        W = torch.diag_embed(weights)
        
        pred_weighted = W @ pred_ori
        sparse_weighted = W @ sparse_ori.unsqueeze(-1)
        return sparse_weighted, pred_weighted
    
    def knn_aligns(self, 
        sparse_disparities: torch.Tensor, 
        pred_disparities: torch.Tensor, 
        sparse_masks: torch.Tensor, 
        complete_masks: torch.Tensor, 
        K: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform K-Nearest Neighbors (KNN) alignment on sparse and predicted disparities.
    
        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            complete_masks (torch.Tensor): Indicating which points in the map to be completed.
            K (int): The number of nearest neighbors to find for each map point.
    
        Returns:
            Tuple: Containing three tensors:
                - dists: The Euclidean distances from each sparse point to its K nearest neighbors.
                - k_sparse_targets: Disparities of the K nearest neighbors from the sparse data.
                - k_pred_targets: Disparities of the K nearest neighbors from the predicted data.
        """
        
        # Coordinates are processed to ensure compatibility with the KNN function.
        batch_sparse = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float() # [N, 3] (b, x, y)
        batch_complete = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float() # [M, 3] (b, x, y)
        
        batch_x, batch_y = batch_sparse[:, 0].contiguous(), batch_complete[:, 0].contiguous()
        x, y = batch_sparse[:, -2:].contiguous(), batch_complete[:, -2:].contiguous()
        
        # Use `torch_cluster.knn` to find K nearest neighbors.
        with torch.cuda.device(self.device):
            knn_map = torch_cluster.knn(x=x, y=y, k=K, batch_x=batch_x, batch_y=batch_y) # [2, M * K]
        knn_indices = knn_map[1, :].view(-1, K)
        
        k_sparse_targets = sparse_disparities[sparse_masks][knn_indices]
        k_pred_targets = pred_disparities[sparse_masks][knn_indices]
        
        knn_coords = x[knn_indices]
        expanded_complete_points = y.unsqueeze(dim=1).repeat(1, K, 1)
        dists = torch.norm(expanded_complete_points - knn_coords, dim=2)
        
        return dists, k_sparse_targets, k_pred_targets
    
    def kss_completer(self, 
        sparse_disparities: torch.Tensor, 
        pred_disparities: torch.Tensor, 
        complete_masks: torch.Tensor, 
        sparse_masks: torch.Tensor, 
        K: int = 5
    ) -> torch.Tensor:
        """
        Perform K-Nearest Neighbors (KNN) interpolation to complete sparse disparities.Use a batch-oriented 
        implementation of KNN interpolation to complete the sparse disparities. We leverages "torch_cluster.knn" 
        for acceleration and GPU memory efficiency.
    
        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map.
            pred_disparities (torch.Tensor): Dredicted disparities for sparse map points.
            complete_masks (torch.Tensor): Indicating which points in the complete map are valid.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            K (int): The number of nearest neighbors to use for interpolation. Defaults to 5.
    
        Returns:
            The completed disparities, interpolated from the nearest neighbors.
        """
        
        # Use `knn_aligns` to find the K nearest neighbors and calculate distances.
        bottomk_dists, k_sparse_targets, k_pred_targets = self.knn_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks, 
            complete_masks=complete_masks,
            K=K
        )
        
        scaled_preds = torch.zeros_like(sparse_disparities, device=self.device, dtype=torch.float32)
        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets, 
            k_pred_targets=k_pred_targets, 
            currk_dists=bottomk_dists, 
            knn=True
        )
        
        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds[complete_masks] = pred_disparities[complete_masks] * scale + shift
        # The completed disparities are computed by combining the scaled predictions and the original sparse disparities.
        scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
        return scaled_preds
    
    def global_aligns(self, 
        sparse_disparities: torch.Tensor, 
        pred_disparities: torch.Tensor, 
        sparse_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform global alignment on sparse and predicted disparities. Extract the valid disparities from 
        both sparse and predicted map based on the sparse masks.
    
        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
    
        Returns:
            Tuple[torch.Tensor]: Containing two tensors:
                - k_sparse_targets: The valid disparities from the sparse map.
                - k_pred_targets: The valid disparities from the predicted map.
        """
        
        # The valid disparities are extracted and unsqueezed to maintain consistent dimensions.
        k_sparse_targets = sparse_disparities[sparse_masks].unsqueeze(dim=0)
        k_pred_targets = pred_disparities[sparse_masks].unsqueeze(dim=0)
        
        return k_sparse_targets, k_pred_targets
    
    def ss_completer(self, 
        sparse_disparities: torch.Tensor, 
        pred_disparities: torch.Tensor, 
        sparse_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Complete sparse disparities using a simple scaling and shifting approach. Perform a global 
        alignment of the sparse and predicted disparities, then applies a scaling and shifting 
        transformation to complete the sparse disparities.
    
        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
    
        Returns:
            The completed disparities, computed by scaling and shifting the predicted disparities.
        """
        
        # Use `global_aligns` to extract valid disparities.
        k_sparse_targets, k_pred_targets = self.global_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks
        )
        
        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets, 
            k_pred_targets=k_pred_targets
        )
        
        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds = pred_disparities * scale + shift
        return scaled_preds
```

### prior_depth_anything/plugin.py

```python
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from PIL import Image

from . import PriorDepthAnything
from .utils import (
    depth2disparity,
    disparity2depth
)

class PriorDARefinerMetrics:
    def __init__(self, align_func=None):
        self.align_func = align_func
    
    def calc_errors(self, gt, pred):
        """Compute metrics for 'pred' compared to 'gt'

        Args:
            gt (torch.Tensor): Ground truth values
            pred (torch.Tensor): Predicted values

            gt.shape should be equal to pred.shape

        Returns:
            dict: Dictionary containing the following metrics:
                'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
                'abs_rel': Absolute relative error
                'rmse': Root mean squared error
        """
        thresh = torch.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        return {k: v.item() for k, v in dict(a1=a1, abs_rel=abs_rel, rmse=rmse).items()}
    
    # Align affine-invariant data to metric data.
    def align_depth_least_square(self, gt, aff, mask, space='depth'):
        """ The input should be in the same size [H, W] or [B, H, W] """
        assert (
            gt.shape == aff.shape == mask.shape
        ), f"{gt.shape}, {aff.shape}, {mask.shape}"
        
        if space == 'depth':
            gt_disparity = depth2disparity(gt)
            aff_disparity = depth2disparity(aff)
        elif space == 'disparity':
            gt_disparity = gt
            aff_disparity = aff
        else:
            raise ValueError("`space` should be in ['depth', 'disparity']")
        
        if len(gt_disparity.shape) == 2:
            gt_disparity = gt_disparity.unsqueeze(0)
            aff_disparity = aff_disparity.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        assert len(gt_disparity.shape) == 3
        
        aligned_disparity = self.align_func(
            sparse_disparities=gt_disparity,
            pred_disparities=aff_disparity,
            sparse_masks=mask
        )
        
        aligned_depth = disparity2depth(aligned_disparity)
        return aligned_depth.squeeze()
    
    def __call__(self, gt_depth, raw_depth, refined_depth):
        gt_mask = gt_depth > 0.0001
        
        raw_depth = self.align_depth_least_square(gt_depth, raw_depth, gt_mask)
        refined_depth = self.align_depth_least_square(gt_depth, refined_depth, gt_mask)
        
        calc_mask = gt_mask
        raw_m = self.calc_errors(gt_depth[calc_mask], raw_depth[calc_mask])
        ref_m = self.calc_errors(gt_depth[calc_mask], refined_depth[calc_mask])
        
        return raw_m, ref_m

class PriorDARefiner(PriorDepthAnything):
    def __init__(self, 
        device: str = 'cuda:0', 
        coarse_only: bool = False, 
        mde_dir: Optional[str] = None, 
        ckpt_dir: Optional[str] = None, 
        frozen_model_size: Optional[str] = None, 
        conditioned_model_size: Optional[str] = None,
        version="1.0"
    ):
        
        super(PriorDARefiner, self).__init__(
            device=device, 
            coarse_only=coarse_only, 
            mde_dir=mde_dir, 
            ckpt_dir=ckpt_dir,
            frozen_model_size=frozen_model_size, 
            conditioned_model_size=conditioned_model_size,
            version=version
        )
        
        self.extra_samples = '500'
        self.metrics_calculater = PriorDARefinerMetrics(align_func=self.completion.ss_completer)
        
        """
        We implement two strategies to filter out low-quality areas in depth_map,
        users can design other filtering methods if neccessary.
        NOTE: For different samples, the sampling strategy could differ for further
        performance improvement.
        """
        self.filter_noisy_depth = {
            'quantile': self.quant_sample,
            'normalization': self.norm_sample
        }
        
    def raw_refined_metrics(self, gt_depth, raw_depth, refined_depth):
        return self.metrics_calculater(gt_depth, raw_depth, refined_depth)
        
    def quant_sample(self, image, depth_map, confidence, quant):
        thres = torch.quantile(confidence, quant)
        extra_depth = depth_map * (confidence < thres).to(torch.float32)
        
        device = depth_map.device
        _, extra_sampled_mask, _ = self.sampler.get_sparse_depth(
            image=image.cpu().numpy(), 
            prior=extra_depth.cpu(), 
            pattern=self.extra_samples
        )
        extra_sampled_mask = extra_sampled_mask.to(device)
        
        sampled = depth_map * ((confidence > thres) | extra_sampled_mask)
        return sampled
    
    def norm_sample(self, image, depth_map, confidence, thres):
        norm_conf = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        extra_depth = depth_map * (norm_conf < thres).to(torch.float32)
        
        device = depth_map.device
        _, extra_sampled_mask, _ = self.sampler.get_sparse_depth(
            image=image.cpu().numpy(), 
            prior=extra_depth.cpu(), 
            pattern=self.extra_samples
        )
        extra_sampled_mask = extra_sampled_mask.to(device)
        
        sampled = depth_map * ((confidence > thres) | extra_sampled_mask)
        return sampled
    
    # Infer one sample once.
    @torch.no_grad()
    def predict(self, 
            image: Union[torch.Tensor, str], # [H, W, 3] in torch.uint8
            depth_map: Union[torch.Tensor, str], # [H, W] in torch.float32
            confidence: Union[torch.Tensor, str], # [H, W] in torch.float32
            thres=0.3 # The `thres` is tunable to obtain better performance.
        ): 
        """ `depth_map` and `confidence` are expected to be on the same deivce. """
        
        # We allow datas to be read locally.
        if isinstance(image, str):
            image = torch.from_numpy(np.asarray(Image.open(image)).astype(np.uint8))
        if isinstance(depth_map, str):
            depth_map = torch.from_numpy(np.asarray(Image.open(depth_map)).astype(np.float32))
        if isinstance(confidence, str):
            confidence = torch.from_numpy(np.asarray(Image.open(confidence)).astype(np.float32))
        h_me, w_me = image.shape[:2]
        
        # The input datas' shape may fit to the output of depth models.
        depth_map = F.interpolate(
            depth_map[None, None, ...], size=(h_me, w_me), mode='bilinear', align_corners=True).squeeze()
        confidence = F.interpolate(
            confidence[None, None, ...], size=(h_me, w_me), mode='bilinear', align_corners=True).squeeze()
        
        # Sample in the pred depth base on the confidence map.
        # We use the combination of two strategies here for more robust outputs.
        keep_mode = ['quantile', 'normalization']
        refineds_with_diff_mode = []
        for md in keep_mode:
            prior = self.filter_noisy_depth[md](image, depth_map, confidence, thres)
            refined = self.infer_one_sample(image=image, prior=prior, geometric=None)
            refineds_with_diff_mode.append(refined)
            
        refined_depth = torch.stack(refineds_with_diff_mode, dim=-1).mean(dim=-1)
        
        return refined_depth, depth_map # return the resized depth_map for evaluation.
    
```

### prior_depth_anything/sparse_sampler.py

```python
import numpy as np
from PIL import Image
import cv2
import re
import warnings

import torch
import torch_cluster
import torch.nn.functional as F

from typing import Dict, Union, Optional

class SparseSampler:
    def __init__(self, device='cuda:0', completion=None):
        self.device = device
        self.min_depth = 0.0001 # We always filter out depth <= 0.
        self.completion = completion

    def __call__(self, 
        image: Union[str, torch.Tensor, np.ndarray], 
        prior: Union[str, torch.Tensor, np.ndarray], 
        geometric: Union[str, torch.Tensor, np.ndarray, None] = None,
        pattern: Optional[str] = None, 
        K: int = 5, 
        prior_cover: bool = False,
        down_fill_mode: str = 'linear'
    ) -> Dict[str, torch.Tensor]:
        """
        1. Handles the loading and preprocessing of image and prior depth data. 
        2. Samples sparse depth points based on the provided pattern or prior depth information.
    
        Args:
            image: 
                The path of the image (readable for Image.open()) or a tensor/array representing the image.
                Shape should be [H, W, 3] with values in the range [0, 255].
            prior: 
                The path of the prior depth (e.g., '*.png') or a tensor/array representing the prior depth.
                Shape should be [H, W] with type float32.
            geometric (optional): 
                The path of the geometric depth (e.g., '*.png') or a tensor/array representing the geometric depth.
                Shape should be [H, W] with type float32.
            pattern (optional): 
                Pattern for sampling sparse depth points. If None, prior depth is used.
            K (int): 
                The minimum number of known points required. Defaults to 5.
            prior_cover (bool, optional): 
                Determine if the prior depth should be used to cover sparse points. Defaults to False.
    
        Returns:
            Dict[str, torch.Tensor]: Containing the processed data, including:
                - 'rgb': The loaded RGB image.
                - 'prior_depth': The loaded prior depth.
                - 'sparse_depth': The sampled sparse depth.
                - 'sparse_mask': Indicating valid sparse depth points.
                - 'cover_mask': Indicating covered points based on prior depth.
        """
        
        assert pattern is None or isinstance(pattern, str)
        data = {}
        
        # Load RGB image.
        if isinstance(image, str):
            if image.endswith('.npy'):
                np_image = np.load(image)
                ts_image = torch.from_numpy(np_image).permute(2, 0, 1).to(torch.uint8)
            else:
                pil_image = Image.open(image)
                np_image = np.asarray(pil_image)
                ts_image = torch.from_numpy(np_image.copy()).permute(2, 0, 1).to(torch.uint8)
        elif isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy()
            ts_image = image.cpu().permute(2, 0, 1).to(torch.uint8)
        elif isinstance(image, np.ndarray):
            np_image = image.copy()
            ts_image = torch.from_numpy(image).permute(2, 0, 1).to(torch.uint8)
        data['rgb'] = ts_image.unsqueeze(0)
        
        # Load prior depth.
        if isinstance(prior, str):
            if prior.endswith('.npy'):
                np_prior = np.load(prior)
                ts_prior = torch.from_numpy(np_prior)
            else:
                # The format should be compatible with Image.open
                pil_prior = Image.open(prior)
                np_prior = np.asarray(pil_prior).astype(np.float32)
                ts_prior = torch.from_numpy(np_prior.copy())
        elif isinstance(prior, np.ndarray):
            ts_prior = torch.from_numpy(prior)
        elif isinstance(prior, torch.Tensor):
            ts_prior = prior.cpu()
        data['prior_depth'] = ts_prior.unsqueeze(0).unsqueeze(0)
        
        # Load geometric depth.
        if geometric is not None:
            if isinstance(geometric, str):
                if geometric.endswith('.npy'):
                    np_geometric = np.load(geometric)
                    ts_geometric = torch.from_numpy(np_geometric)
                else:
                    # The format should be compatible with Image.open
                    pil_geometric = Image.open(geometric)
                    np_geometric = np.asarray(pil_geometric).astype(np.float32)
                    ts_geometric = torch.from_numpy(np_geometric.copy())
            elif isinstance(geometric, np.ndarray):
                ts_geometric = torch.from_numpy(geometric)
            elif isinstance(geometric, torch.Tensor):
                ts_geometric = geometric.cpu()
            data['geometric_depth'] = ts_geometric.unsqueeze(0).unsqueeze(0)
        
        # Sample the points manually if `pattern` is provided, otherwise use prior.
        if pattern or ts_prior.shape[-2:] != ts_image.shape[-2:]:
            sparse_depth, sparse_mask, cover_mask = self.get_sparse_depth(
                image=np_image, prior=ts_prior, pattern=pattern, down_fill_mode=down_fill_mode
            )
            
            # We do not implement hybrid-pattern here.
            if ts_prior.shape[-2:] != ts_image.shape[-2:]:
                assert pattern is None, "When testing with low-res prior, please set `pattern` to None"
            
            """ Force to keep the prior in the condition. """
            if prior_cover:
                assert ts_prior.shape[-2:] == ts_image.shape[-2:]
                cover_mask = ts_prior > self.min_depth
        else:
            """
            If `pattern` is None, the value of `prior_cover` does not 
            matter and all prior will cover in kss_completer.    
            """
            sparse_depth = ts_prior.clone()
            sparse_mask = sparse_depth > self.min_depth
            cover_mask = torch.zeros_like(sparse_mask)
            
        data['sparse_depth'] = sparse_depth.unsqueeze(0).unsqueeze(0)
        data['sparse_mask'] = sparse_mask.unsqueeze(0).unsqueeze(0)
        data['cover_mask'] = cover_mask.unsqueeze(0).unsqueeze(0)
        
        # Check samples and move points to the target device.
        if sparse_mask.sum() < K:
            raise ValueError("There are not enough known points.")
        data = {k: v.to(self.device) for k, v in data.items() if v is not None}
        return data
    
    def get_sparse_depth(self, image, prior, pattern=None, down_fill_mode='linear'):
        height, width, c = image.shape[-3:]
        low_height, low_width = prior.shape[-2:]
        
        if height != low_height or width != low_width:
            pattern = 'downscale_'
            # print("============================ Testing with known low depth. ============================")
        # else:
        #     print(f"============================ Testing with {pattern}. ============================")
        
        if pattern.isdigit():
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            num_sample = int(pattern)
            
            idx_nnz = torch.nonzero(prior.view(-1) > self.min_depth, as_tuple=False)
            num_idx = len(idx_nnz)
            if num_idx < num_sample:
                warnings.warn(
                    f"Aiming to sample {num_sample} points, but only {num_idx} valid points in the map.")
                
            idx_sample = torch.randperm(num_idx)[:num_sample]
            idx_nnz = idx_nnz[idx_sample[:]]
            
            sparse_mask = torch.zeros((height * width), dtype=torch.bool)
            sparse_mask[idx_nnz] = True
            sparse_mask = sparse_mask.view((height, width))
            
            sparse_depth = prior * sparse_mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)
            
        elif re.fullmatch(r'^downscale_\d*$', pattern):
            prior = prior.unsqueeze(0)
            
            if pattern != 'downscale_':
                prior_mask = prior > self.min_depth
                
                factor = pattern.split("_")[-1]
                factor = int(factor)
            
                # Fill in the blank areas in the image before downsampling.
                if down_fill_mode == 'linear':
                    filled_depth = self.linear_interpolate_depths(
                        sparse_depths=prior, 
                        sparse_masks=prior_mask, 
                        complete_masks=~prior_mask
                    )
                else: # if down_fill_mode == 'global' or 'knn'
                    inter_sparse_depth, inter_sparse_mask, inter_cover_mask = self.get_sparse_depth(
                        image=image, prior=prior, pattern='2000', down_fill_mode=down_fill_mode
                    )
                    filled_depth = self.inpaint_interpolate_depths(
                        image=image, 
                        prior_depth=inter_sparse_depth, 
                        prior_mask=inter_sparse_mask,
                        ret=down_fill_mode
                    )
                
                
                # Downscale the prior depth map.
                low_height, low_width = height // factor, width // factor
                prior = F.interpolate(
                    filled_depth.unsqueeze(0), 
                    size=(low_height, low_width), 
                    mode='bilinear', 
                    align_corners=True
                )
                prior = prior.squeeze()
            
            # Insert the low-res prior depth map into the higher one.
            s_height, s_width = height / low_height, width / low_width
            idx_height = (s_height * torch.arange(low_height)).long()
            idx_width = (s_width * torch.arange(low_width)).long()

            down_mask = torch.zeros((height, width), dtype=torch.bool)
            down_mask[..., idx_height[:, None], idx_width] = True
            
            sparse_depth = torch.zeros((height, width), dtype=torch.float32)
            sparse_depth[down_mask] = prior.flatten()
            
            sparse_mask = sparse_depth > self.min_depth
            # Filter the sparse mask with valid mask if sampled manually.
            if pattern != 'downscale_': sparse_mask &= prior_mask.squeeze(0)
            sparse_depth = sparse_depth * sparse_mask.type_as(sparse_depth)
            cover_mask = torch.zeros_like(sparse_mask)
            
        elif re.fullmatch(r'^cubic_\d+$', pattern):
            clen = pattern.split('_')[-1]
            clen = int(clen)
            
            # Sample a cube in the image based on top-lerf coords and clen
            cubic_mask = torch.ones_like(prior, dtype=torch.bool)
            height_upper, width_upper = height - clen, width - clen
            h = np.random.randint(0, height_upper)
            w = np.random.randint(0, width_upper)
            cubic_mask[h : h+clen, w : w+clen] = False
            cover_mask = torch.logical_and(cubic_mask, prior > self.min_depth)
            
            vacant_depth = prior * cover_mask.type_as(prior)
            sparse_depth, sparse_mask, _ = self.get_sparse_depth(image, vacant_depth, pattern='2000')
            
        elif re.fullmatch(r'^distance_\d+_\d+$', pattern):
            # The lower bound and high-bound of the interval that 
            # we want to keep the depth known
            low_dist, high_dist = pattern.split('_')[-2:]
            low_dist, high_dist = int(low_dist), int(high_dist)
            
            # Only keep depth within the range --- (low_dist, high_dist)
            cover_mask = torch.logical_and(
                (prior > self.min_depth),
                torch.logical_and(
                    prior > low_dist, 
                    prior < high_dist
                )
            )
            
            range_depth = prior * cover_mask.type_as(prior)
            sparse_depth, sparse_mask, _ = self.get_sparse_depth(image, range_depth, pattern='2000')
            
        elif pattern == 'sift' or pattern == 'orb':
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            assert image is not None
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if pattern == "sift":
                detector = cv2.SIFT.create()
            elif pattern == "orb":
                detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            else:
                raise NotImplementedError

            keypoints = detector.detect(gray)
            mask = torch.zeros([height, width])

            if len(keypoints) < 20:
                return self.get_sparse_depth(image=image, prior=prior, pattern='2000')
            
            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                mask[x, y] = 1.0

            train_sfm_max_dropout_rate = 0.0
            if train_sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, train_sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(mask)
                mask_keep = torch.bernoulli(mask_keep)

                mask = mask * mask_keep

            sparse_mask = (mask * (prior > self.min_depth).type_as(prior)).to(torch.bool)
            sparse_depth = prior * mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)
            
        elif re.fullmatch(r'^LiDAR_\d+$', pattern):
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            w_c = 0.5 * width
            h_c = 0.5 * height
            focal = height

            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            dep_np = prior.numpy()

            # sample the lidar patterns
            pitch_max = 0.5
            pitch_min = -0.5
            num_lines = int(pattern.split('_')[1])
            num_horizontal_points = 200

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi / 2.1, np.pi / 2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid)  # assume the distace is unit
            z = np.sqrt(1. - x ** 2 - y ** 2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1)  # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8)  # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[:, ((points_2D[0] >= 0) & (points_2D[0] < width) & (
                        points_2D[1] >= 0) & (points_2D[1] < height))]

            mask = np.zeros([height, width])
            mask[points_2D_valid[1], points_2D_valid[0]] = 1.0
            # only keep the pred_disparitiesorginal valid regions
            mask = mask * (dep_np > self.min_depth).astype(float)
            
            sparse_mask = torch.from_numpy(mask).to(torch.bool)
            sparse_depth = prior * sparse_mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)
            
        else:
            raise NotImplementedError((
                "'pattern' should be in format of ['^LiDAR_\d+$', 'sift', "
                "'orb', '^cubic_\d+$', '^distance_\d+_\d+$'," 
                "'^downscale_\d*$', '(int)'], but the provided 'pattern' is -- '{}'".format(pattern)
            ))
        
        return sparse_depth, sparse_mask, cover_mask
    
    
    def inpaint_interpolate_depths(self, image, prior_depth, prior_mask, ret='scaled'):
        from .utils import disparity2depth, log_img
        
        ori_device = 'cpu'
        ts_image = torch.from_numpy(image.copy()).permute(2, 0, 1).to(torch.uint8)
        
        ts_image = ts_image.unsqueeze(0).to(self.device)
        prior_depth = prior_depth.unsqueeze(0).to(self.device)
        prior_mask = prior_mask.unsqueeze(0).to(self.device)
        
        completed_maps = self.completion(
            images=ts_image, 
            sparse_depths=prior_depth, 
            sparse_masks=prior_mask, 
            ret=ret
        )
        
        filled_depth = disparity2depth(completed_maps)
        return filled_depth.to(ori_device)
        
    
    def linear_interpolate_depths(self, sparse_depths, sparse_masks, complete_masks):
        known_points = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float() # [N, 3] (b, x, y)
        complete_depths = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float() # [M, 3] (b, x, y)
        
        batch_x, batch_y = known_points[:, 0].contiguous(), complete_depths[:, 0].contiguous()
        x, y = known_points[:, -2:].contiguous(), complete_depths[:, -2:].contiguous()
        
        knn_map = torch_cluster.knn(x=x, y=y, k=5, batch_x=batch_x, batch_y=batch_y) # [2, M * K]
        knn_indices = knn_map[1, :].view(-1, 5)
        knn_depths = sparse_depths[sparse_masks][knn_indices]
        
        filled_depths = torch.zeros_like(sparse_depths)
        filled_depths[sparse_masks] = sparse_depths[sparse_masks]
        filled_depths[complete_masks] = knn_depths.mean(dim=-1)
        
        return filled_depths
```

### prior_depth_anything/utils.py

```python
import torch
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from typing import Tuple
import matplotlib

@dataclass
class Arguments:
    K: int = field(
        default=5, 
        metadata={"help": "K value of KNN"}
    )
    conditioned_model_size: str = field(
        default="vitb", 
        metadata={"help": "Size of conditioned model."}
    )
    frozen_model_size: str = field(
        default="vitb", 
        metadata={"help": "Size of frozen model."}
    )
    normalize_depth: bool = field(
        default=True, 
        metadata={"help": "Whether to normalize depth."}
    )
    normalize_confidence: bool = field(
        default=True, 
        metadata={"help": "Whether to normalize confidence."}
    )
    double_global: bool = field(
        default=False, 
        metadata={"help": "Whether to use double globally-aligned conditions."}
    )

    repo_name: str = field(
        default='Rain729/Prior-Depth-Anything', metadata={"help": "Name of hf-repo."})
    log_dir: str = field(
        default='output', metadata={"help": "The root path to save visualization results."})
    # down_fill_mode: str = field(
    #     default='linear', 
    #     metadata={
    #         "help": (
    #             "The mode to fill in the vacancy in the prior. Only works for `pattern='^downscale_\d*$'`. "
    #             "Choices=('knn', 'global', 'linear')"
    #         )
    #     }
    # )
    
    
# ******************** disparity space ********************
# Adapted from Marigold, available at https://github.com/prs-eth/Marigold
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
# ************************* end ****************************
    
    
def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize_depth_maps(
        depth_map, 
        min_depth, 
        max_depth, 
        cmap="Spectral", 
        valid_mask=None
    ):
        """
        Colorize depth maps.
        """
        assert len(depth_map.shape) >= 2, "Invalid dimension"

        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().clone().squeeze().numpy()
        elif isinstance(depth_map, np.ndarray):
            depth = depth_map.copy().squeeze()
        # reshape to [ (B,) H, W ]
        if depth.ndim < 3:
            depth = depth[np.newaxis, :, :]

        # colorize
        cm = matplotlib.colormaps[cmap]
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
        img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
        img_colored_np = np.rollaxis(img_colored_np, 3, 1)

        if valid_mask is not None:
            if isinstance(depth_map, torch.Tensor):
                valid_mask = valid_mask.detach().numpy()
            valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
            if valid_mask.ndim < 3:
                valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
            else:
                valid_mask = valid_mask[:, np.newaxis, :, :]
            valid_mask = np.repeat(valid_mask, 3, axis=1)
            img_colored_np[~valid_mask] = 0

        if isinstance(depth_map, torch.Tensor):
            img_colored = torch.from_numpy(img_colored_np).float()
        elif isinstance(depth_map, np.ndarray):
            img_colored = img_colored_np

        return img_colored
        
def log_img(image, path, valids=None, scale=None, shift=None):
    if valids is not None:
        invalids = ~valids
        image[invalids] = 0
        
    if scale is None:
        scale, shift = image.max() - image.min(), image.min()
    
    normalized_value = (image - shift) / scale
    if "error" in path: normalized_value = 1 - normalized_value
    value_colored = colorize_depth_maps(
        normalized_value, 0, 1, cmap="Spectral"
    ).squeeze()
    
    if valids is not None:
        invalids = np.repeat(~valids[None, ...], 3, axis=0)
        value_colored[invalids] = 0
    value_colored = (value_colored * 255).astype(np.uint8)
    value_colored = Image.fromarray(chw2hwc(value_colored))
    value_colored.save(path)
```

