import torch
import numpy as np

from PIL import Image

def save_vis(arr, pth, is_normal=False, valid_mask=None, gt=None):
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        arr[~valid_mask] = arr.min()
    if gt is not None:
        arr = (arr - gt.min()) / (gt.max() - gt.min())
    elif is_normal:
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            arr = arr.detach().cpu()
        arr = arr.numpy()

    arr = np.clip(arr, a_min=0, a_max=1)
    if arr.max() <= 1:
        arr = arr * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(pth)