"""
Foreground/background extraction using SAM
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.special import softmax as np_softmax
from tqdm import tqdm

import torchvision.transforms.v2 as tfms
import pandas as pd
from scripts.baselines.seg.lang_sam import LangSAM

def get_dataframe(data):
    df = pd.DataFrame(data)
    return df


def generate_masks_langsam(mask_gen, dataset, dataframes_dir=None, vis=False):
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

    df_data = []

    Path(dataset.mask_dir).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        img_pil, label, meta = dataset[i]
        img_path = meta['image_path']
        mask_path = dataset.im2mask_path(img_path)

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
            mask = Image.fromarray(np.zeros_like(np.array(img)))
            mask.save(mask_path)
            extra = {'error': str(e)}
        extra['image_path'] = img_path
        df_data.append(extra)

    if dataframes_dir is not None:
        df = get_dataframe(df_data)
        df.to_hdf(os.path.join(dataframes_dir, f'masks_info.h5'), key='df', mode='w')


def generate_masks_sam(mask_gen, dataset):
    # TODO unify with langsam!!!
    # crate path for mask including parents if it does not exist
    Path(dataset.mask_dir).mkdir(parents=True, exist_ok=True)

    idxs = np.arange(len(dataset))
    for idx in tqdm(idxs):
        im, label, meta = dataset[idx]

        mask_path = dataset.im2mask_path(meta['image_path'])
        # if mask exists, skip
        if os.path.exists(mask_path):
            continue

        # convert im from torch [c, h, w] to numpy array [h, w, c]
        im = (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        mask = mask_gen.predict_batch([im])

        # PIL saves images with lossless compression so good enough for us -
        # multiply by 255 for easier visual inspection
        mask = Image.fromarray(mask * 255)
        mask.save(mask_path)


def get_mask_generator(cfg):
    if cfg.mask_gen.model == 'sam':
        return SAMMaskGenerator(
            ckpt_path=cfg.path.ckpt,
            method=cfg.mask_gen.method,
        )
    elif cfg.mask_gen.model == 'lang_sam':
        if cfg.mask_gen.method == 'metaclass':
            text_prompt = f'{cfg.data.metaclasses[0]}'
        else:
            text_prompt = f'{cfg.mask_gen.text_prompt}'
        return LangSAMMaskGenerator(
            ckpt_path=cfg.path.ckpt,
            text_prompt=text_prompt,
            dataframes_dir=cfg.mask_gen.dataframes_dir,
            method=cfg.mask_gen.method,
        )
    else:
        raise ValueError(f"Unknown mask generator {cfg.mask_gen.model}")


@hydra.main(config_path="conf", config_name=get_config(mode='generate_masks'))
def main(cfg):
    splits = [cfg.mask_gen.split] if cfg.mask_gen.split != 'all' else ['val', 'test', 'train']

    mask_gen = get_mask_generator(cfg)

    generate_masks = generate_masks_sam if cfg.mask_gen.model == 'SAM' else generate_masks_langsam

    cfg.data.mask_subdir = mask_gen.name

    for split in splits:

        dataset = get_datasets(cfg, splits=[split])

        generate_masks(
            mask_gen=mask_gen,
            dataset=dataset[split],
        )


def test_generate_masks_langsam():
    cfg = get_mock_config('fungitastic', root='.', mode='generate_masks')
    split = 'val'

    mask_gen = get_mask_generator(cfg)

    generate_masks = generate_masks_sam if cfg.mask_gen.model == 'SAM' else generate_masks_langsam

    cfg.data.mask_subdir = mask_gen.name

    datasets = get_datasets(cfg, splits=[split], check_mask_dir=False)

    dataframes_dir = datasets[split].mask_dir

    generate_masks(
        mask_gen=mask_gen,
        dataset=datasets[split],
        dataframes_dir=dataframes_dir,
    )


if __name__ == '__main__':
    # test_generate_masks_langsam()
    main()