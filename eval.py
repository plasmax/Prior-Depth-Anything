from torch.utils.data import DataLoader
from nyu_full_res import NYU_FULL_RES
from prior_depth_anything import PriorDepthAnything
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    data_val = NYU_FULL_RES(mode="eval")
    test_loader=DataLoader(
        dataset=data_val, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    priorda = PriorDepthAnything(
        device=device,
        postfix='_1_1',
        extra_condition='spmask',
        fmde_dir='/home/chensiyu/ckpts/Prior-Depth-Anything',
        ckpt_dir='/home/chensiyu/ckpts/Prior-Depth-Anything'
    )

    abs_rels = []

    pbar = tqdm(enumerate(test_loader), desc=f"Loop: Validation", total=len(test_loader))
    for idx, batch_data in pbar:
        pred_depth = priorda.infer_one_sample(
            image=batch_data['image_path'][0], 
            prior=batch_data['depth_path'][0],
            pattern="500",
            down_fill_mode="linear"
        )

        gt_depth = np.asarray(
            Image.open(
                batch_data['depth_path'][0]
            )
        ).astype(np.float32)
        calc_gt = torch.from_numpy(gt_depth).to(pred_depth.device)
        calc_mask = calc_gt > 0.
        pred_depth = pred_depth.squeeze()
        metrics = priorda.calc_errors(calc_gt[calc_mask], pred_depth[calc_mask])
        abs_rels.append(metrics['abs_rel'])
        pbar.set_description(f"ABS_REL: {metrics['abs_rel']}")

    print("=========== Final mean results. ===============")
    print(np.mean(abs_rels))
