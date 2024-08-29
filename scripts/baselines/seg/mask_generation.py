from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import os
from tqdm import tqdm

from dataset.fungi import FungiTastic
from mask_generator import LangSAMMaskGenerator

import pandas as pd


def im2mask_path(im_path, mask_dir, img_dir):
    path_wo_img_dir = os.path.relpath(im_path, img_dir)
    mask_path = os.path.join(mask_dir, path_wo_img_dir)
    return mask_path


def generate_masks(mask_gen, dataset, mask_dir, dataframes_dir=None, vis=False):
    """
    Mask generation using LangSAM: https://github.com/luca-medeiros/lang-segment-anything
    Requires having groundingdino installed: https://github.com/IDEA-Research/GroundingDINO

    dataset should be a PyTorch dataset, with dataset[2] being meta, where meta['image_path'] must exist.
    dataset should have a method im2mask_path(self, im_path) -> mask_path
    sam_ckpt_h is the path to the SAM model checkpoint.
    text_prompt is the prompt to use for LangSAM.
    if dataframes_dir is not None, it will save the dataframes to that directory.
    """
    assert hasattr(dataset, 'im2mask_path'), "Dataset must have a method im2mask_path(self, im_path) -> mask_path"
    assert 'image_path' in dataset[0][2], "meta['image_path'] must exist in the dataset"

    img_dir = dataset[0][2]['image_path']

    df_data = []

    Path(dataset.mask_dir).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        img_pil, label, meta = dataset[i]
        img_path = meta['image_path']
        mask_path = im2mask_path(img_path, mask_dir, img_dir)

        # make sure img_pil is actually a PIL image
        if not isinstance(img_pil, Image.Image):
            img_pil = Image.fromarray(img_pil)

        # TODO add config param to overwrite existing masks
        if os.path.exists(mask_path):
            continue

        try:
            # TODO rewrite to batch version
            mask, extra = mask_gen.predict(img_pil)

            mask = Image.fromarray(mask * 255)

            if vis:
                plt.subplot(1, 2, 1)
                plt.imshow(img_pil)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(mask)
                plt.axis('off')

                plt.show()

            mask.save(mask_path)
            extra['error']: None

        except Exception as e:
            print(e)
            print(f"Error with image {img_path}")
            # save empty mask, add error to extra
            mask = Image.fromarray(np.zeros_like(np.array(img_pil)))
            mask.save(mask_path)
            extra = {'error': str(e)}
        extra['image_path'] = img_path
        df_data.append(extra)

    if dataframes_dir is not None:
        df = pd.DataFrame(df_data)
        df.to_hdf(os.path.join(dataframes_dir, f'masks_info.h5'), key='df', mode='w')


def get_mask_generator(cfg):
    text_prompt = 'mushroom'
    return LangSAMMaskGenerator(
        ckpt_path=cfg.ckpt_path,
        text_prompt=text_prompt,
        dataframes_dir=cfg.mask_gen.dataframes_dir,
    )


def main():
    with open('../../../config/path.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    split = 'val'

    mask_gen = get_mask_generator(cfg)

    dataset = FungiTastic(
        root=cfg.data_path,
        split=split,
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    generate_masks(
        mask_gen=mask_gen,
        dataset=dataset,
        mask_dir='TODO'
    )


if __name__ == '__main__':
    main()