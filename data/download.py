import os
import subprocess

import zipfile
from pathlib import Path
import argparse

SUBSET2SIZES = {
    'all': ['300', '500'],
    'fs': ['500'],
    'm': ['300', '500', '720', 'fullsize'],
    'dna': ['300', '500', '720', 'fullsize']
}

SUBSET2STR = {
    'fs': 'FewShot',
    'm': 'Mini',
    'dna': 'DNA'
}

SUBSETS = SUBSET2SIZES.keys()

DOWNLOAD_ROOT = 'https://cmp.felk.cvut.cz/datagrid/personal/janoukl1/shared_ext/FungiTastic/'

def download_file(url: str, target_path: str) -> None:
    """
    Download a file from a given URL to a target path using wget.

    Args:
        url (str): URL of the file to download.
        target_path (str): Local path where the file should be saved.
    """
    subprocess.run(["wget", "-nc", "-P", target_path, url])
    target_file = os.path.join(target_path, os.path.basename(url))
    assert os.path.exists(target_file), ("Something is wrong with the filename creation logic or the download url, file"
                                         f" {target_path} does not exist")


def args2img_link(subset: str, size: str) -> str:
    """
    Generate the download link for images based on the provided subset and size.

    Args:
        subset (str): Subset of the dataset.
        size (str): Size of the images.

    Returns:
        str: Download link for the images.
    """
    size_str = args.size + 'p' if not args.size == 'fullsize' else 'fullsize'
    if args.subset != 'all':
        return f"{DOWNLOAD_ROOT}/FungiTastic-{SUBSET2STR[args.subset]}-{size_str}.zip"
    else:
        return f"{DOWNLOAD_ROOT}/FungiTastic-{size_str}.zip"


def download(subset: str, size: str, data_root: str, rewrite: bool = False, keep_zip: bool = False, no_extraction: bool = False) -> None:
    """
    Download and optionally extract the FungiTastic dataset and metadata.

    Args:
        subset (str): Subset of the dataset to download.
        size (str): Size of the images to download.
        data_root (str): Root directory for downloading data.
        rewrite (bool, optional): Rewrite existing files. Defaults to False.
        keep_zip (bool, optional): Keep the downloaded zip files. Defaults to False.
        no_extraction (bool, optional): Do not extract the downloaded zip files. Defaults to False.
    """
    validate_args(subset, size, data_root)
    fungi_path = os.path.join(data_root, 'FungiTastic')
    Path(fungi_path).mkdir(parents=False, exist_ok=True)

    metadata_link = f"{DOWNLOAD_ROOT}/metadata.zip"
    metadata_folder = os.path.join(fungi_path, 'metadata')
    Path(metadata_folder).mkdir(parents=False, exist_ok=True)
    metadata_path = os.path.join(metadata_folder, 'metadata.zip')
    if not os.path.exists(metadata_path) or rewrite:
        print(f"Downloading metadata from {metadata_link} to {metadata_folder}")
        download_file(metadata_link, metadata_folder)
        print(f"Download complete")

    fungi_img_link = args2img_link(subset, size)
    filename = os.path.basename(fungi_img_link)
    fungi_img_path = os.path.join(fungi_path, filename)
    if not os.path.exists(fungi_img_path) or rewrite:
        print(f"Downloading images from {fungi_img_link} to {fungi_path}")
        download_file(fungi_img_link, fungi_path)
        print(f"Download complete")
    else:
        print(f"Images already downloaded to {fungi_img_path}")

    if not no_extraction:
        # Unzip images and metadata
        with zipfile.ZipFile(metadata_path, 'r') as zip_ref:
            zip_ref.extractall(fungi_path)
        print(f"Unzipped metadata to {metadata_path}")

        with zipfile.ZipFile(fungi_img_path, 'r') as zip_ref:
            zip_ref.extractall(fungi_path)
        print(f"Unzipped images to {fungi_path}")

    if not keep_zip:
        os.remove(fungi_img_path)
        print(f"Removed {fungi_img_path}")
        os.remove(metadata_path)
        print(f"Removed {metadata_path}")


def validate_args(args):
    """
    Validate the provided command line arguments.
    :param args: Command line arguments.
    """
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    if args.subset not in SUBSETS:
        raise ValueError(f"Invalid subset: {args.subset}")

    if args.size not in SUBSET2SIZES[args.subset]:
        raise ValueError(f"Invalid size for subset {args.subset}: {args.size}. "
                         f"Available sizes are: {SUBSET2SIZES[args.subset]}")


def get_args():
    """
    Parse and return the command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='fs', choices=SUBSETS)
    parser.add_argument('--size', type=str, default='300')
    parser.add_argument('--data_root', type=str, default='your_datasets_path')
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--keep_zip', action='store_true')
    parser.add_argument('--no_extraction', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    validate_args(args)
    download(args.subset, args.size, args.data_root, args.rewrite, args.keep_zip, args.no_extraction)
