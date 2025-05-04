from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional, Union
from types import SimpleNamespace

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
from fgvc.datasets import ImageDataset
# Use seaborn style for better visualization
from matplotlib import style

style.use("seaborn-v0_8-whitegrid")


class FungiTastic(ImageDataset):
    """
    Dataset class for the Danish Fungi dataset.

    Dataframe keys: ['eventDate', 'year', 'month', 'day', 'habitat', 'countryCode',
                     'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family',
                     'genus', 'specificEpithet', 'hasCoordinate', 'species',
                     'iucnRedListCategory', 'substrate', 'latitude', 'longitude',
                     'coorUncert', 'observationID', 'region', 'district', 'filename',
                     'category_id', 'metaSubstrate', 'poisonous', 'elevation', 'landcover',
                     'biogeographicalRegion', 'image_path']
    """
    SUBSET2SIZES: Dict[str, List[str]] = {
        'all': ['300', '500'],
        'FewShot': ['300', '500'],
        'Mini': ['300', '500', '720', 'fullsize'],
    }

    SUBSET2TASKS: Dict[str, List[str]] = {
        'all': ['open', 'closed'],
        'FewShot': ['closed'],
        'Mini': ['open', 'closed'],
    }

    SUBSET2SPLITS: Dict[str, List[str]] = {
        'all': ['train', 'val', 'test', 'dna'],
        'FewShot': ['train', 'val', 'test'],
        'Mini': ['train', 'val', 'test', 'dna'],
    }

    SPLIT2STR: Dict[str, str] = {
        'train': 'Train',
        'val': 'Val',
        'test': 'Test',
        'dna': 'DNA-Test'
    }

    TASK2STR: Dict[str, str] = {
        'open': 'OpenSet',
        'closed': 'ClosedSet',
    }

    SUBSETS = SUBSET2SIZES.keys()

    def __init__(self, root: str, data_subset: str = 'Mini', split: str = 'val', size: str = '300',
                 task: str = 'closed', transform: T.Compose = None, **kwargs):
        df = self.get_df(
            data_path=root,
            split=split,
            size=size,
            task=task,
            data_subset=data_subset
        )

        assert "image_path" in df
        self.df = df
        self.transform = transform
        self.data_subset = data_subset
        self.split = split
        self.task = task
        self.img_root = root

        if split in ['train', 'val']:
            assert "category_id" in df
            category_id2label = df.groupby('category_id')['species'].unique().to_dict()
            unknown_species = list(category_id2label.get(-1, []))
            self.unkwnown_id = -1
            self.category_id2label = {k: v[0] for k, v in category_id2label.items()}
            self.label2category_id = {v: k for k, v in self.category_id2label.items()}
            category_id2label[self.unkwnown_id] = unknown_species
            for unk_spec in unknown_species:
                self.label2category_id[unk_spec] = self.unkwnown_id

            self.n_classes = len(self.df['category_id'].unique())

    def get_class_id(self, idx: int) -> int:
        """
        Get class id of i-th element in the dataset.

        Args:
            idx (int): Index of the element.

        Returns:
            int: Class ID of the element.
        """
        return self.df["category_id"].iloc[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int], str]:
        """
        Get item (image, class ID, and file path) by index.

        Args:
            idx (int): Index of the element.

        Returns:
            Tuple[torch.Tensor, Optional[int], str]: Image tensor, class ID, and file path.
        """
        if self.split in ['train', 'val']:
            return super().__getitem__(idx)
        else:
            image, file_path = self.get_image(idx)
            image = self.apply_transforms(image)
            return image, None, file_path

    @staticmethod
    def check_params(data_subset: str, split: str, size: str, task: str) -> None:
        """
        Check the validity of dataset parameters.

        Args:
            data_subset (str): Subset of the dataset.
            split (str): Data split.
            size (str): Image size.
            task (str): Task type.

        Raises:
            AssertionError: If any parameter is invalid.
        """
        assert data_subset in FungiTastic.SUBSETS, f"Invalid subset: {data_subset}. Available subsets are: {FungiTastic.SUBSETS}"
        assert split in ['train', 'val', 'test',
                         'dna'], f"Invalid split: {split}. Available splits are: ['train', 'val', 'test', 'dna']"
        assert size in FungiTastic.SUBSET2SIZES[data_subset], (f"Invalid size: {size}. Available sizes for subset "
                                                               f"{data_subset} are: {FungiTastic.SUBSET2SIZES[data_subset]}")
        assert task in FungiTastic.SUBSET2TASKS[data_subset], (f"Invalid task: {task}. Available tasks for subset "
                                                               f"{data_subset} are: {FungiTastic.SUBSET2TASKS[data_subset]}")
        assert split in FungiTastic.SUBSET2SPLITS[data_subset], (f"Invalid split: {split}. Available splits for subset "
                                                                 f"{data_subset} are: {FungiTastic.SUBSET2SPLITS[data_subset]}")
        assert not (task == 'open' and split == 'dna'), "Open set task is not available for DNA split"

    @staticmethod
    def get_df(data_path: str, split: str = 'val', task: str = 'closed', size: str = '300',
               data_subset: str = 'Mini') -> pd.DataFrame:
        """
        Get the dataframe for the specified dataset parameters.

        Args:
            data_path (str): Path to the dataset.
            split (str): Data split.
            task (str): Task type.
            size (str): Image size.
            data_subset (str): Subset of the dataset.

        Returns:
            pd.DataFrame: Dataframe containing the dataset metadata.
        """
        FungiTastic.check_params(data_subset, split, size, task)

        subfolder_str = f'FungiTastic-{data_subset}' if data_subset != 'all' else 'FungiTastic'
        data_subset_str = f'-{data_subset}' if data_subset != 'all' else ''
        task_str = f'-{FungiTastic.TASK2STR[task]}' if (data_subset != 'FewShot' and split != 'train') else ''

        df_path = os.path.join(
            data_path,
            "metadata",
            subfolder_str,
            f"FungiTastic{data_subset_str}{task_str}-{FungiTastic.SPLIT2STR[split]}.csv",
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, subfolder_str, split, f'{size}p', x)
        )
        return df

    def show_sample(self, idx: int) -> None:
        """
        Display a sample image with its class name and ID.

        Args:
            idx (int): Index of the sample to display.
        """
        image, category_id, file_path = self.__getitem__(idx)
        class_name = self.category_id2label[category_id] if category_id is not None else '[TEST]'
        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()

    def get_category_idxs(self, category_id: int) -> List[int]:
        """
        Get all indexes of a specific class ID.

        Args:
            category_id (int): Class ID to search for.

        Returns:
            List[int]: List of indexes that belong to the specified class ID.
        """
        return self.df[self.df.category_id == category_id].index.tolist()


if __name__ == '__main__':
    # Use LaTeX-like font for paper visualization if needed and LaTeX is installed
    if False:  # Change to True if you want LaTeX-like font
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    with open('../config/path.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    dataset = FungiTastic(
        root=cfg.data_path,
        split='val',
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
    )
    dataset.show_sample(1)
