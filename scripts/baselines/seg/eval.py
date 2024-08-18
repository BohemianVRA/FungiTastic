import os

import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm
from PIL import Image

import numpy as np
from types import SimpleNamespace
from dataset.mask_fungi import MaskFungiTastic
import torch

from torchmetrics.functional import jaccard_index

def evaluate_saved_masks(dataset, masks_path, debug=False, vis=False, thresh=0.5):
    ious = []
    idxs = np.arange(len(dataset)) if not debug else np.random.choice(len(dataset), 10)
    for idx in tqdm(idxs):
        image, gt_mask, class_id, file_path = dataset[idx]
        mask_path = os.path.join(masks_path, os.path.basename(file_path))
        pred_mask = Image.open(mask_path)

        # resize pred_mask to gt_mask size
        pred_mask = pred_mask.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)

        iou = jaccard_index(
            preds=(torch.tensor(np.array(pred_mask)) / 255).int(),
            target=torch.tensor(gt_mask),
            task='binary',
            threshold=thresh
        )
        ious.append(iou)

        if vis:
            #  show image, gt_mask and pred_mask
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title(f"Class: {dataset.category_id2label[class_id]}")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask)
            plt.title(f"GT Mask")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask)
            plt.title(f"Pred Mask")
            plt.axis('off')
            plt.suptitle(f"ID: {idx}; IoU: {iou}")
            plt.show()

    ious = np.array(ious)
    iou_all = ious.mean()
    print(f"IoU: {iou_all}")

    # plot the iou histogram and show the mean and stds
    plt.hist(ious, bins=20)
    plt.axvline(iou_all, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all + ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all - ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.show()


    print()




def main():
    split = 'val'
    with open('../../../config/path.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    dataset = MaskFungiTastic(
        root=cfg.data_path,
        split=split,
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
    )

    evaluate_saved_masks(
        dataset=dataset,
        masks_path='/mnt/vrg2/imdec/masks/FungiTastic/lang_sam_metaclass/FungiTastic-Mini/300p/')


if __name__ == '__main__':
    main()