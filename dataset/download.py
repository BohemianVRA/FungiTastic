import argparse
import subprocess
import zipfile
from pathlib import Path


class FungiTasticDownloader:
    """
    A downloader class for the FungiTastic dataset, providing options to download and extract
    various components including images, metadata, satellite data, climatic data, and masks.

    Attributes:
        SUBSETS (list): Available dataset subsets.
        SIZES (list): Available image sizes.
        SUBSET2STR (dict): Mapping of subset abbreviations to full names.
        HAS_DNA (dict): Indicates which subsets have DNA data.
        DOWNLOAD_ROOT (str): Root URL for dataset downloads.

    Methods:
        download_file(url, tool): Downloads a file using wget or curl.
        download_and_extract(url, target_dir): Downloads and extracts a zip file.
        download_metadata(): Downloads the metadata zip file.
        download_images(subset, size): Downloads image data for a given subset and size.
        download_satellite_data(): Downloads satellite data.
        download_climatic_data(): Downloads climatic data.
        download_masks(): Downloads mask data.
        generate_img_link(subset, size, split): Generates download URL for images.
        download(subset, size): Initiates download of selected data types.
        validate_params(params): Validates input arguments for download.
    """

    SUBSETS = ["full", "fs", "m"]
    SIZES = ["300", "500", "720", "fullsize"]
    SUBSET2STR = {"fs": "FewShot", "m": "Mini"}
    HAS_DNA = {
        "full": True,
        "fs": False,
        "m": True,
    }
    DOWNLOAD_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"

    def __init__(
        self,
        save_path: Path,
        rewrite: bool = False,
        keep_zip: bool = False,
        no_extraction: bool = False,
        metadata: bool = False,
        images: bool = False,
        satellite: bool = False,
        climatic: bool = False,
        masks: bool = False,
    ):
        """
        Initializes the downloader with user-specified parameters.

        Args:
            save_path (Path): Root directory to save the downloaded data.
            rewrite (bool): Whether to overwrite existing files.
            keep_zip (bool): Whether to keep the downloaded zip files.
            no_extraction (bool): If True, skip the extraction step.
            metadata (bool): If True, download metadata.
            images (bool): If True, download images.
            satellite (bool): If True, download satellite data.
            climatic (bool): If True, download climatic data.
            masks (bool): If True, download mask data.
        """
        self.save_path = save_path
        self.rewrite = rewrite
        self.keep_zip = keep_zip
        self.no_extraction = no_extraction
        self.metadata = metadata
        self.images = images
        self.satellite = satellite
        self.climatic = climatic
        self.masks = masks

        self.fungi_path = self.save_path / "FungiTastic"
        self.fungi_path.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, tool: str = "wget") -> Path:
        """
        Downloads a zip file from a specified URL using wget or curl.

        Args:
            url (str): The URL of the file to download.
            tool (str): The command-line tool to use for downloading ('wget' or 'curl').

        Returns:
            Path: Path to the downloaded file.

        Raises:
            RuntimeError: If the file download fails.
        """
        target_file = self.fungi_path / Path(url).name
        cmd = (
            ["wget", "-nc", "-P", str(self.fungi_path), url]
            if tool == "wget"
            else ["curl", "-O", str(target_file), url]
        )

        print(f"Downloading {url} to {target_file}")
        print(f"Command: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0 or not target_file.exists():
            raise RuntimeError(f"Failed to download {url}")

        return target_file

    def download_and_extract(self, url: str, target_dir: Path) -> None:
        """
        Downloads and extracts a zip file from the specified URL.

        Args:
            url (str): The URL of the zip file.
            target_dir (Path): Directory to extract the zip file to.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / Path(url).name

        print("Download:")
        if self.rewrite or not zip_path.exists():
            print(f"\tDownloading from {url} to {target_dir}")
            zip_file = self.download_file(url)
            print(f"\tDownload of {url} complete\n")
        else:
            zip_file = zip_path
            print(f"\tFile already downloaded to {zip_file}\n")

        print("Extract:")
        if not self.no_extraction:
            print(f"\tExtracting {zip_file}")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"\tUnzipped to {target_dir}\n")
        else:
            print(f"\tSkipping extraction of {zip_file}\n")

        print("Cleanup:")
        if not (self.keep_zip or self.no_extraction):
            print(f"\tRemoving the zip file {zip_file}")
            zip_file.unlink()
            print(f"\tRemoved {zip_file}")
        else:
            print(f"\tKeeping the zip file {zip_file}\n")

    def download_metadata(self) -> None:
        """Downloads and extracts the metadata zip file."""
        metadata_url = f"{self.DOWNLOAD_ROOT}/metadata.zip"
        self.download_and_extract(metadata_url, self.fungi_path / "metadata")

    def download_images(self, subset: str, size: str) -> None:
        """
        Downloads image data for a given subset and size.

        Args:
            subset (str): The subset of images to download.
            size (str): The size of the images.
        """
        splits = ["train", "val", "test"] + (
            ["dna-test"] if self.HAS_DNA[subset] else []
        )
        for split in splits:
            img_link = self.generate_img_link(subset=subset, size=size, split=split)
            self.download_and_extract(img_link, self.fungi_path)

    def download_satellite_data(self) -> None:
        """Downloads and extracts satellite data (NIR and RGB)."""
        satellite_files = ["satellite_NIR.zip", "satellite_RGB.zip"]
        for file in satellite_files:
            satellite_link = f"{self.DOWNLOAD_ROOT}/{file}"
            self.download_and_extract(satellite_link, self.fungi_path)

    def download_climatic_data(self) -> None:
        """Downloads and extracts climatic data."""
        climatic_url = f"{self.DOWNLOAD_ROOT}/climatic.zip"
        self.download_and_extract(climatic_url, self.fungi_path)

    def download_masks(self) -> None:
        """Downloads and extracts mask data."""
        masks_url = f"{self.DOWNLOAD_ROOT}/masks.zip"
        self.download_and_extract(masks_url, self.fungi_path)

    def generate_img_link(self, subset: str, size: str, split: str) -> str:
        """
        Generates the download link for image data based on subset, size, and split.

        Args:
            subset (str): Subset of the dataset.
            size (str): Image size.
            split (str): Data split (train, val, test, dna-test).

        Returns:
            str: The generated URL for the image download.
        """
        size_str = f"{size}p" if size != "fullsize" else "fullsize"
        if subset != "full":
            return f"{self.DOWNLOAD_ROOT}/FungiTastic-{self.SUBSET2STR.get(subset, '')}-{split}-{size_str}.zip"
        else:
            return f"{self.DOWNLOAD_ROOT}/FungiTastic-{split}-{size_str}.zip"

    def download(self, subset: str = None, size: str = None) -> None:
        """
        Downloads selected components of the FungiTastic dataset.

        Args:
            subset (str): Subset of the dataset to download (optional).
            size (str): Image size to download (optional).
        """
        if self.metadata:
            self.download_metadata()

        if self.images:
            self.download_images(subset, size)

        if self.satellite:
            self.download_satellite_data()

        if self.climatic:
            self.download_climatic_data()

        if self.masks:
            self.download_masks()

    @staticmethod
    def validate_params(params) -> None:
        """
        Validates the input parameters to ensure correctness.

        Args:
            params: Parsed input arguments.

        Raises:
            FileNotFoundError: If the specified save_path does not exist.
            ValueError: If invalid subset, size, or other input arguments are provided.
        """
        save_path = Path(params.save_path)
        if not save_path.exists():
            raise FileNotFoundError(f"Data root not found: {save_path}")

        if params.images and not (params.subset and params.size):
            raise ValueError("Subset and size must be provided to download images.")

        if params.subset and params.subset not in FungiTasticDownloader.SUBSETS:
            raise ValueError(f"Invalid subset: {params.subset}")

        if params.size and params.size not in FungiTasticDownloader.SIZES:
            raise ValueError(
                f"Invalid size: {params.size}. Available sizes are: {', '.join(FungiTasticDownloader.SIZES)}"
            )


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for downloading the FungiTastic dataset.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Download FungiTastic dataset.")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Root directory for saving datasets.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=FungiTasticDownloader.SUBSETS,
        help="Subset of the dataset to download.",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=FungiTasticDownloader.SIZES,
        help="Size of the images to download.",
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
    parser.add_argument(
        "--climatic", action="store_true", help="Download climatic data."
    )
    parser.add_argument(
        "--satellite", action="store_true", help="Download satellite data."
    )
    parser.add_argument("--masks", action="store_true", help="Download masks.")
    parser.add_argument("--images", action="store_true", help="Download images.")
    parser.add_argument("--metadata", action="store_true", help="Download metadata.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    FungiTasticDownloader.validate_params(args)

    downloader = FungiTasticDownloader(
        save_path=Path(args.save_path),
        rewrite=args.rewrite,
        keep_zip=args.keep_zip,
        no_extraction=args.no_extraction,
        metadata=args.metadata,
        images=args.images,
        satellite=args.satellite,
        climatic=args.climatic,
        masks=args.masks,
    )

    downloader.download(args.subset, args.size)
