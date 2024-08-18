import ast
import os
from typing import Tuple, Any
import yaml
import numpy as np
from types import SimpleNamespace


import torch
import torchvision.transforms as T

import pandas as pd
from dataset.fungi import FungiTastic


class MaskFungiTastic(FungiTastic):
    def __init__(self, root: str, data_subset: str = 'Mini', split: str = 'val', size: str = '300',
                 task: str = 'closed', transform: T.Compose = None, **kwargs):
        super().__init__(
            root=root,
            data_subset=data_subset,
            split=split,
            size=size,
            task=task,
            transform=transform,
            **kwargs
        )
        # load mask_test.csv
        gt_masks = pd.read_csv(os.path.join(root, 'masks_test2.csv'))
        # gt_masks = pd.read_parquet(os.path.join(root, 'mask_test.parquet'))

        gt_masks.rename(columns={'file_name': 'filename'}, inplace=True)
        gt_masks['rle'] = gt_masks['rle'].apply(ast.literal_eval)
        self.df = self.df.merge(gt_masks, on='filename', how='inner')


    def rle_to_mask(self, rle_points: list, height: int, width: int) -> np.ndarray:
        """Decode data compressed with CVAT-based Run-Length Encoding (RLE) and return image mask.

        Parameters
        ----------
        rle_points
            List of points stored in shape with mask type returned by CVAT API.
        height
            Height of image.
        width
            Width of image.

        Returns
        -------
        mask
            A 2D NumPy array that represent image mask.
        """
        """
            Decode a Run-Length Encoding (RLE) into a 2D NumPy array (image mask).

            Parameters
            ----------
            rle_points : list
                A list of points representing the RLE encoding of the mask.
            height : int
                The height of the image mask.
            width : int
                The width of the image mask.

            Returns
            -------
            mask : np.ndarray
                A 2D NumPy array representing the decoded image mask.
            """
        # Initialize an empty mask
        mask = np.zeros(height * width, dtype=np.uint8)

        # Extract the RLE encoding
        rle_counts = rle_points[:-4]  # Exclude the last four points which are bounding box coordinates

        # Decode the RLE into the mask
        current_position = 0
        current_value = 0

        for rle_count in rle_counts:
            mask[current_position:current_position + rle_count] = current_value
            current_position += rle_count
            current_value = 1 - current_value  # Toggle between 0 and 1

        # Reshape the flat mask back to 2D
        mask = mask.reshape((height, width))

        return mask

    def __getitem__(self, item):
        image, class_id, file_path = super().__getitem__(item)

        meta = self.df.iloc[item]
        mask_rle = meta['rle']

        mask = self.rle_to_mask(mask_rle, meta.height, meta.width)

        return image, mask, class_id, file_path

    def show_sample(self, idx: int) -> None:
        """
        Display a sample image with its class name and ID.

        Args:
            idx (int): Index of the sample to display.
        """
        image, mask, category_id, file_path = self.__getitem__(idx)
        class_name = self.category_id2label[category_id] if category_id is not None else '[TEST]'
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask)

        plt.title(f"Mask: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Use LaTeX-like font for paper visualization if needed and LaTeX is installed
    if False:  # Change to True if you want LaTeX-like font
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    split = 'val'
    with open('../config/path.yaml', "r") as f:
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

    for i in range(20, 25):
        dataset.show_sample(i)
    # dataset.show_sample(25)
    # print(dataset.df.iloc[25])


