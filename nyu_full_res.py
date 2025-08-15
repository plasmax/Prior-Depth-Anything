import os
import tarfile
from io import BytesIO

import numpy as np

from PIL import Image
import torch
import h5py
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path


split_txt = "./filename_list_test.txt"

class NYU_FULL_RES:
    def __init__(self, mode):

        self.mode = mode

        # Only support testing for now
        if mode != 'eval' and mode != 'test':
            raise NotImplementedError

        self.min_depth, self.max_depth = 0.0001, 10.
        
        print('Loading NYU Full Res...')
        with open(split_txt, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data_dir = str(Path('/home/chensiyu/prida_data/') / "marigold" / "nyuv2")
        rgb_rel_path, _, filled_depth_rel_path = self.filenames[idx]
        
        return {
            "image_path": os.path.join(data_dir, rgb_rel_path),
            "depth_path": os.path.join(data_dir, filled_depth_rel_path),
        }