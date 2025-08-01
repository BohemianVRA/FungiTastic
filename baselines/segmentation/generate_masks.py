from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import os
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from dataset.fungi import FungiTastic
from mask_generator import GDINOSAM
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def im2mask_path(im_path, mask_dir, img_dir):
    """
    Convert an image path to its corresponding mask path.
    
    Args:
        im_path (str): Path to the input image
        mask_dir (str): Directory where masks should be saved
        img_dir (str): Root directory of the images
        
    Returns:
        str: Path where the mask should be saved
    """
    path_wo_img_dir = os.path.relpath(im_path, img_dir)
    mask_path = os.path.join(mask_dir, path_wo_img_dir)
    return mask_path


def generate_masks(mask_gen, dataset, mask_dir, dataframes_dir=None, vis=False):
    """
    Generate segmentation masks for all images in the dataset using the provided mask generator.
    
    Args:
        mask_gen: Mask generator model (e.g. GDINOSAM)
        dataset: Dataset containing images to process
        mask_dir (str): Directory to save generated masks
        dataframes_dir (str, optional): Directory to save metadata about generated masks
        vis (bool): Whether to visualize the generated masks
    """
    df_data = []

    for i in tqdm(range(len(dataset))):
        img_pil, label, img_path = dataset[i]
        mask_path = im2mask_path(img_path, mask_dir, dataset.img_root)

        # if os.path.exists(mask_path):
        #     print(f"Mask already exists for {mask_path}")
        #     continue

        # Create mask directory structure if it doesn't exist
        if i == 0:
            print(f"Creating mask directory {mask_path}")
            Path(os.path.dirname(mask_path)).mkdir(parents=True, exist_ok=True)

        # Ensure input is PIL Image
        if not isinstance(img_pil, Image.Image):
            img_pil = Image.fromarray(img_pil)
            
        # Generate mask using the model
        mask, extra = mask_gen.predict(img_pil)

        # Convert mask to PIL Image and scale to 0-255 range
        mask = Image.fromarray(mask * 255)

        # Optional visualization
        if vis:
            plt.subplot(1, 2, 1)
            plt.imshow(img_pil)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.axis('off')

            plt.show()

        # Save mask and collect metadata
        mask.save(mask_path)
        extra['error']: None
        extra['image_path'] = img_path
        df_data.append(extra)

    # Save metadata if requested
    if dataframes_dir is not None:
        df = pd.DataFrame(df_data)
        df.to_hdf(os.path.join(dataframes_dir, f'masks_info.h5'), key='df', mode='w')


def get_mask_generator(cfg):
    """
    Initialize and return a mask generator model.
    
    Args:
        cfg: Configuration object containing model parameters
        
    Returns:
        GDINOSAM: Initialized mask generator model
    """
    text_prompt = 'mushroom'
    return GDINOSAM(
        ckpt_path=cfg.ckpt_path,
        text_prompt=text_prompt,
    )


def main(cfg):
    """
    Main function to generate masks for the fungi dataset.
    
    Args:
        cfg: Configuration object containing dataset and model parameters
    """
    # Initialize mask generator
    mask_gen = get_mask_generator(cfg)

    # Load dataset
    dataset = FungiTastic(
        root=cfg.data_path,
        split=cfg.split,
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
    )

    # Generate masks
    generate_masks(
        mask_gen=mask_gen,
        dataset=dataset,
        mask_dir=cfg.mask_path
    )


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate masks for fungi dataset')
    parser.add_argument('--config_path', type=str, default=os.path.join(SCRIPT_DIR, 'config/seg.yaml'))
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    main(cfg)