import argparse
import subprocess
import zipfile
from pathlib import Path

SUBSETS = ["full", "fs", "m", "dna"]
SIZES = ["300", "500", "720p", "fullsize"]

SUBSET2STR = {"fs": "FewShot", "m": "Mini", "dna": "DNA"}

DOWNLOAD_ROOT = (
    "https://cmp.felk.cvut.cz/datagrid/personal/janoukl1/shared_ext/FungiTastic/"
)


def download_file(url: str, target_dir: Path, tool: str = "wget") -> Path:
    """
    Download a file from a given URL to a target directory using the specified download tool.

    Args:
        url (str): URL of the file to download.
        target_dir (Path): Directory where the file should be saved.
        tool (str): The download tool to use ('wget' or 'curl'). Defaults to 'wget'.

    Returns:
        Path: Path to the downloaded file.

    Raises:
        RuntimeError: If the file download fails.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / Path(url).name

    if tool == "wget":
        result = subprocess.run(["wget", "-nc", "-P", str(target_dir), url])
    elif tool == "curl":
        result = subprocess.run(["curl", "-O", str(target_file), url])
    else:
        raise ValueError("Unsupported download tool. Use 'wget' or 'curl'.")

    if result.returncode != 0 or not target_file.exists():
        raise RuntimeError(f"Failed to download {url}")

    return target_file


def generate_img_link(subset: str, size: str) -> str:
    """
    Generate the download link for images based on the provided subset and size.

    Args:
        subset (str): Subset of the dataset.
        size (str): Size of the images.

    Returns:
        str: Download link for the images.
    """
    size_str = size + "p" if size != "fullsize" else "fullsize"
    if subset != "full":
        return f"{DOWNLOAD_ROOT}/FungiTastic-{SUBSET2STR[subset]}-{size_str}.zip"
    else:
        return f"{DOWNLOAD_ROOT}/FungiTastic-{size_str}.zip"


def download_dataset(
    subset: str,
    size: str,
    save_path: Path,
    rewrite: bool = False,
    keep_zip: bool = False,
    no_extraction: bool = False,
) -> None:
    """
    Download and optionally extract the FungiTastic dataset and metadata.

    Args:
        subset (str): Subset of the dataset to download.
        size (str): Size of the images to download.
        save_path (Path): Root directory for downloading dataset.
        rewrite (bool, optional): Rewrite existing files. Defaults to False.
        keep_zip (bool, optional): Keep the downloaded zip files. Defaults to False.
        no_extraction (bool, optional): Do not extract the downloaded zip files. Defaults to False.
    """
    fungi_path = save_path / "FungiTastic"
    fungi_path.mkdir(parents=True, exist_ok=True)

    # Download and extract metadata
    metadata_link = f"{DOWNLOAD_ROOT}/metadata.zip"
    metadata_folder = fungi_path / "metadata"
    metadata_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_folder / "metadata.zip"

    if not metadata_path.exists() or rewrite:
        print(f"Downloading metadata from {metadata_link} to {metadata_folder}")
        metadata_zip = download_file(metadata_link, metadata_folder)
        print("Download complete")
    else:
        metadata_zip = metadata_path
        print(f"Metadata already downloaded to {metadata_zip}")

    if not no_extraction:
        with zipfile.ZipFile(metadata_zip, "r") as zip_ref:
            zip_ref.extractall(fungi_path)
        print(f"Unzipped metadata to {fungi_path}")

    # Download and extract images
    img_link = generate_img_link(subset, size)
    img_zip_path = fungi_path / Path(img_link).name

    if not img_zip_path.exists() or rewrite:
        print(f"Downloading images from {img_link} to {fungi_path}")
        img_zip = download_file(img_link, fungi_path)
        print("Download complete")
    else:
        img_zip = img_zip_path
        print(f"Images already downloaded to {img_zip}")

    if not no_extraction:
        with zipfile.ZipFile(img_zip, "r") as zip_ref:
            zip_ref.extractall(fungi_path)
        print(f"Unzipped images to {fungi_path}")

    if not keep_zip:
        img_zip.unlink()
        print(f"Removed {img_zip}")
        metadata_zip.unlink()
        print(f"Removed {metadata_zip}")


def validate_args(args) -> None:
    """
    Validate the provided command line arguments.

    Args:
        args: Command line arguments.

    Raises:
        FileNotFoundError: If the specified data root directory does not exist.
        ValueError: If the subset or size is invalid.
    """
    save_path = Path(args.save_path)
    if not save_path.exists():
        raise FileNotFoundError(f"Data root not found: {save_path}")

    if args.subset not in SUBSETS:
        raise ValueError(f"Invalid subset: {args.subset}")

    if args.size not in SIZES:
        raise ValueError(
            f"Invalid size for subset {args.subset}: {args.size}. "
            f"Available sizes are: {', '.join(SIZES)}"
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Download FungiTastic dataset.")
    parser.add_argument(
        "--subset",
        type=str,
        default="fs",
        choices=SUBSETS,
        help="Subset of the dataset to download.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="300",
        choices=SIZES,
        help="Size of the images to download.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="your_datasets_path",
        help="Root directory for saving datasets.",
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="Rewrite existing files."
    )
    parser.add_argument(
        "--keep_zip", action="store_true", help="Keep the downloaded zip files."
    )
    parser.add_argument(
        "--no_extraction",
        action="store_true",
        help="Do not extract the downloaded zip files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    validate_args(args)
    download_dataset(
        args.subset,
        args.size,
        Path(args.save_path),
        args.rewrite,
        args.keep_zip,
        args.no_extraction,
    )
