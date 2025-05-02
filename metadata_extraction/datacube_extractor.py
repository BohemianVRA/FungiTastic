"""
This module provides classes for extracting and processing data from raster files (like GeoTIFFs).
It includes functionality for both data extraction and image generation from raster data.
"""

import os.path as osp
import os

import numpy as np
import pyproj
import rasterio
from PIL import Image


class DataCubeExtractor:
    """
    A class for extracting data from raster files at specific geographic coordinates.
    
    This class handles the conversion between geographic coordinates (latitude/longitude)
    and raster coordinates, allowing for data extraction at specific points or regions.
    
    Attributes:
        tile_bbox (tuple): Bounding box coordinates for the tile (left, bottom, width, height)
        band_index (int): Index of the raster band to extract data from
        padding (float): Padding around the tile in degrees
        lat_column (str): Name of the latitude column in input data
        lon_column (str): Name of the longitude column in input data
        transformer (pyproj.Transformer): Coordinate transformer between WGS84 and raster CRS
        tile_data (numpy.ndarray): The loaded raster data for the tile
        x_resolution (float): Resolution of the raster in x direction
        y_resolution (float): Resolution of the raster in y direction
    """

    def __init__(
        self,
        tile_bbox: tuple,
        band_index: int = 1,
        padding: float = 1,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ):
        """
        Initialize the DataCubeExtractor.

        Args:
            tile_bbox (tuple): Bounding box coordinates (left, bottom, width, height) in degrees
            band_index (int, optional): Index of the raster band to extract. Defaults to 1.
            padding (float, optional): Padding around the tile in degrees. Defaults to 1.
            lat_column (str, optional): Name of the latitude column. Defaults to "latitude".
            lon_column (str, optional): Name of the longitude column. Defaults to "longitude".
        """
        self.tile_bbox = tile_bbox
        self.band_index = band_index
        self.padding = padding
        self.lat_column = lat_column
        self.lon_column = lon_column

        # From loaded raster
        self.transformer = None
        self.tile_data = None

        self.x_resolution, self.y_resolution = None, None
        self.left, self.bottom = None, None
        self.right, self.top = None, None

    def load_raster(self, raster_path: str) -> bool:
        """
        Load raster data for the specified tile from a GeoTIFF file.

        Args:
            raster_path (str): Path to the GeoTIFF file

        Returns:
            bool: True if the tile contains data, False otherwise
        """

        with rasterio.open(raster_path, "r") as tif_file:

            meta = tif_file.meta
            meta.update(
                count=tif_file.count
            )  # update the count of the meta to match the number of layers

            self.transformer = pyproj.Transformer.from_crs(
                "epsg:4326", tif_file.crs, always_xy=True
            )  # xx = lon, yy = lat

            self.x_resolution, self.y_resolution = tif_file.res[:2]

            self.left, self.bottom = self.transformer.transform(
                yy=self.tile_bbox[0] - self.padding, xx=self.tile_bbox[1] - self.padding
            )
            self.right, self.top = self.transformer.transform(
                yy=self.tile_bbox[0] + self.tile_bbox[2] + self.padding,
                xx=self.tile_bbox[1] + self.tile_bbox[3] + self.padding,
            )

            tile_window = rasterio.windows.from_bounds(
                self.left,
                self.bottom,
                self.right,
                self.top,
                transform=tif_file.transform,
            )
            self.tile_data = tif_file.read(self.band_index, window=tile_window)

            return self.tile_data.shape[0] > 0 and self.tile_data.shape[1] > 0

    def __getitem__(self, item: tuple[float, float]):
        """
        Extract data value at the specified geographic coordinates.

        Args:
            item (tuple[float, float]): Geographic coordinates (latitude, longitude)

        Returns:
            float or None: The data value at the specified coordinates, or None if out of bounds
        """

        # convert the lat, lon coordinates to raster EPSG.
        x_index, y_index = self._item_to_tile_index(
            item
        )  # x: left->right, y: top->bottom

        if (x_index and y_index) and (
            y_index <= self.tile_data.shape[0] and x_index <= self.tile_data.shape[1]
        ):
            if self.tile_data[y_index, x_index] == 0:
                return self._search_radius_mean_indexed(
                    x_index, y_index, radius_index=1
                )
            data_point_value = int(self.tile_data[y_index, x_index])

            return data_point_value if data_point_value > -1000 else None

        else:
            print(f"Item {item} not in tile {self.tile_bbox}. Increase padding?")
            return None

    def _item_to_tile_index(self, item: tuple[float, float]):
        """
        Convert geographic coordinates to raster indices.

        Args:
            item (tuple[float, float]): Geographic coordinates (latitude, longitude)

        Returns:
            tuple[int, int] or tuple[None, None]: Raster indices (x, y) or (None, None) if out of bounds
        """
        item_x, item_y = self.transformer.transform(
            yy=item[self.lat_column], xx=item[self.lon_column]
        )

        if (self.left <= item_x <= self.right) and (self.bottom <= item_y <= self.top):
            x_index = int((item_x - self.left) / self.x_resolution)
            y_index = int((item_y - self.bottom) / self.y_resolution
            )

            return x_index, y_index
        return None, None

    def _search_radius_mean_indexed(
        self, x_index: int, y_index: int, radius_index: int = 1
    ) -> float:
        """
        Calculate mean value in a radius around the specified index.

        Args:
            x_index (int): X coordinate in raster space
            y_index (int): Y coordinate in raster space
            radius_index (int, optional): Search radius in pixels. Defaults to 1.

        Returns:
            float: Mean value in the search radius, or 0 if no valid data found
        """
        try:
            return self.tile_data[
                y_index - radius_index : y_index + radius_index,
                x_index - radius_index : x_index + radius_index,
            ].mean()
        except Exception as e:
            print(e)
            return 0


class ImageDataCubeExtractor(DataCubeExtractor):
    """
    A specialized DataCubeExtractor for generating image patches from raster data.
    
    This class extends DataCubeExtractor to add functionality for converting raster data
    to image format and saving image patches around specific points of interest.
    
    Attributes:
        convert_to_uint8 (bool): Whether to convert data to 8-bit unsigned integers
        gamma_for_conversion (float): Gamma correction value for image conversion
        image_patch_size (int): Size of the image patches to generate
    """

    def __init__(
        self,
        tile_bbox: tuple,
        band_index: int = 1,
        padding: float = 2.0,
        convert_to_uint8: bool = True,
        gamma_for_conversion: float = 2.5,
        image_patch_size: int = 128,
    ):
        """
        Initialize the ImageDataCubeExtractor.

        Args:
            tile_bbox (tuple): Bounding box coordinates (left, bottom, width, height) in degrees
            band_index (int, optional): Index of the raster band to extract. Defaults to 1.
            padding (float, optional): Padding around the tile in degrees. Defaults to 2.0.
            convert_to_uint8 (bool, optional): Whether to convert to 8-bit images. Defaults to True.
            gamma_for_conversion (float, optional): Gamma correction value. Defaults to 2.5.
            image_patch_size (int, optional): Size of image patches in pixels. Defaults to 128.
        """
        super().__init__(tile_bbox, band_index=band_index, padding=padding)

        self.convert_to_uint8 = convert_to_uint8
        self.gamma_for_conversion = gamma_for_conversion
        self.image_patch_size = image_patch_size

    def convert_tile_data_to_uint8(self, tile_data: np.ndarray) -> np.ndarray:
        """
        Convert raster data to 8-bit unsigned integer format for image generation.
        
        The conversion process includes:
        1. Normalizing values to 0-1 range
        2. Applying gamma correction
        3. Scaling to 0-255 range
        4. Converting to uint8

        Args:
            tile_data (np.ndarray): Input raster data

        Returns:
            np.ndarray: 8-bit unsigned integer image data
        """
        tile_data = np.clip(tile_data / 10000.0, a_min=0, a_max=1.0)
        tile_data = (tile_data ** (1 / self.gamma_for_conversion)) * 256
        tile_data = tile_data.astype(np.uint8)
        # (tile_data - tile_data.min()) / (tile_data.max() - tile_data.min()) * 255
        return tile_data

    def save_tile_image(self, tile_image_output_dir: str) -> None:
        """
        Save the entire tile as an image file.

        Args:
            tile_image_output_dir (str): Directory to save the image file

        Raises:
            AssertionError: If output directory doesn't exist or tile data isn't loaded
        """
        assert osp.isdir(
            tile_image_output_dir
        ), f"Photo output directory '{tile_image_output_dir}' does not exist!"
        assert self.tile_data is not None, "Tile data not loaded!"

        start_latitude = self.tile_bbox[0]
        start_longitude = self.tile_bbox[1]

        tile_data = self.tile_data.astype(np.uint8)

        image = Image.fromarray(
            tile_data,
            "L",
        )

        # PlotID = int(row['PlotID'])
        # lat = row['lat']
        # lon = row['lon']
        #
        # # construct the outup file path of the patch as './CD/AB/XXXXABCD.jpeg'
        # path = output_path
        # for d in (str(PlotID)[-2:], str(PlotID)[-4:-2]):
        #     path = os.path.join(path, d)
        #     if not os.path.exists(path):
        #         os.makedirs(path)

        image_path = osp.join(
            tile_image_output_dir,
            f"lat-({start_latitude:+03.1f})_lon-({start_longitude:+03.1f}).jpeg",
        )
        image.save(image_path)

    def save_patch_image(self, item, tile_image_output_dir: str) -> str:
        """
        Save an image patch centered on the specified item's location.

        Args:
            item: Data point containing location information
            tile_image_output_dir (str): Directory to save the image patch

        Returns:
            str: Path to the saved image file

        Raises:
            AssertionError: If tile data isn't loaded
        """

        assert self.tile_data is not None, "Tile data not loaded!"

        index_x, index_y = self._item_to_tile_index(item)
        patch_tile = self._select_image_patch(index_x, index_y, self.image_patch_size)

        if self.convert_to_uint8:
            patch_tile = self.convert_tile_data_to_uint8(patch_tile)

        image = Image.fromarray(
            patch_tile,
            "L",
        )

        image_name = f"{item.observationID}.jpeg"

        image_path = tile_image_output_dir + "/" + image_name
        image.save(image_path, "JPEG", quality=100)

        return image_path

    def _select_image_patch(
        self, index_x: int, index_y: int, patch_size: int
    ) -> np.ndarray:
        """
        Extract an image patch centered on the specified coordinates.

        Args:
            index_x (int): X coordinate in raster space
            index_y (int): Y coordinate in raster space
            patch_size (int): Desired size of the patch in pixels

        Returns:
            np.ndarray: Image patch data
        """
        left_x = max(index_x - patch_size // 2, 0)
        right_x = min(index_x + patch_size // 2, self.tile_data.shape[1])
        top_y = max(index_y - patch_size // 2, 0)
        bottom_y = min(index_y + patch_size // 2, self.tile_data.shape[0])

        patch_tile = self.tile_data[top_y:bottom_y, left_x:right_x]

        if (
            self.image_patch_size != patch_tile.shape[0]
            or self.image_patch_size != patch_tile.shape[1]
        ):
            print(
                f"Cropped image {patch_tile.shape} does not match patch size {self.image_patch_size}! Increase padding?"
            )

        return patch_tile
