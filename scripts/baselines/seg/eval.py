import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# get root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm
from PIL import Image

import numpy as np
from types import SimpleNamespace
from dataset.mask_fungi import MaskFungiTastic
import torch

from torchmetrics.functional import jaccard_index


def evaluate_single_image(idx, dataset, masks_path, thresh, vis):
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

    if vis:
        # show image, gt_mask and pred_mask
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

    return iou.item()


def evaluate_saved_masks(dataset, masks_path, debug=False, vis=False, thresh=0.5, result_dir=None, chunk_size=10):
    ious = []
    idxs = np.arange(len(dataset)) if not debug else np.random.choice(len(dataset), 10)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in tqdm(range(0, len(idxs), chunk_size)):
            chunk = idxs[i:i + chunk_size]
            futures = [executor.submit(evaluate_single_image, idx, dataset, masks_path, thresh, vis) for idx in chunk]
            for future in as_completed(futures):
                ious.append(future.result())

    ious = np.array(ious)
    iou_all = ious.mean()
    print(f"IoU: {iou_all}")

    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        # save all as npy
        np.save(result_dir / 'ious.npy', ious)
        # save the mean and std
        with open(result_dir / 'iou.txt', 'w') as f:
            f.write(f"IoU: {iou_all:.2f} +/- {ious.std():.2f}")

    # plot the iou histogram and show the mean and stds
    plt.hist(ious, bins=20)
    plt.axvline(iou_all, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all + ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all - ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.title(f"IoU: {iou_all:.2f} +/- {ious.std():.2f}")
    plt.legend(['mean', 'std'])
    if result_dir is not None:
        plt.savefig(result_dir / 'iou_hist.png')
    plt.show()


def main():
    split = 'val'
    with open('../../../config/FungiTastic_seg.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    result_dir = Path(cfg.path_out) / 'results' / 'seg' / split

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
        masks_path='/mnt/vrg2/imdec/masks/FungiTastic/lang_sam_metaclass/FungiTastic-Mini/300p/',
        result_dir=result_dir,
    )

def visualize():
    def get_data(datast, mask_dir, idx):
        image, gt_mask, class_id, file_path = dataset[idx]
        mask_path = os.path.join(mask_dir, os.path.basename(file_path))
        pred_mask = Image.open(mask_path)

        # resize pred_mask to gt_mask size
        pred_mask = pred_mask.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)

        return image, gt_mask, pred_mask
    config_path = '../../../config/FungiTastic_FS.yaml'
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    split = 'val'
    with open('../../../config/FungiTastic_seg.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    result_dir = Path(cfg.path_out) / 'results' / 'seg' / split

    dataset = MaskFungiTastic(
        root=cfg.data_path,
        split=split,
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
    )

    masks_path = '/mnt/vrg2/imdec/masks/FungiTastic/lang_sam_metaclass/FungiTastic-Mini/300p/'

    # show the first image, iou histogram, gt and predicted mask
    ious_path = result_dir / 'ious.npy'
    ious = np.load(ious_path)

    img, gt, pred = get_data(dataset, masks_path, 3)

    # plt.figure(figsize=(5.5, 7))
    plt.figure(figsize=(11, 3.5))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.hist(ious, bins=20, color='darkblue')
    plt.title('Per-Image IoU Histogram')
    # mean and std
    iou_all = ious.mean()
    plt.axvline(iou_all, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(iou_all + ious.std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(iou_all - ious.std(), color='g', linestyle='dashed', linewidth=2)
    plt.legend(['mean', 'std'])

    plt.subplot(1, 4, 3)
    plt.imshow(gt, cmap='gray')
    plt.imshow(np.zeros_like(gt), cmap='gray', alpha=0.5)
    plt.title('GT Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred)
    plt.imshow(np.zeros_like(pred), cmap='gray', alpha=0.5)
    plt.title('Pred Mask')
    plt.axis('off')

    #  save
    plt.savefig(result_dir / 'rebuttal_masks.png')
    plt.show()



if __name__ == '__main__':
    # main()
    visualize()
