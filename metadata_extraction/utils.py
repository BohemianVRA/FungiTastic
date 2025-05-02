"""
Utility functions for metadata extraction and processing.

This module provides helper functions for:
1. Date handling for quarterly data extraction
2. Creating and managing geographic tiles
3. Extracting data from raster files at specific locations
"""

from typing import Any

import numpy as np
import pandas as pd
from datacube_extractor import DataCubeExtractor, ImageDataCubeExtractor


def get_quarter_dates(year: str, quarter: str) -> str:
    """
    Generate date range string for a specific quarter of a year.
    
    This function is used to create date range strings for ecodatacube data extraction.
    The quarters are defined as:
    - Q1: Dec 2 (prev year) to Mar 20
    - Q2: Mar 21 to Jun 24
    - Q3: Jun 25 to Sep 12
    - Q4: Sep 13 to Dec 1

    Args:
        year (str): The year to generate dates for
        quarter (str): The quarter number (1-4)

    Returns:
        str: Date range string in format "YYYY.MM.DD..YYYY.MM.DD"
    """
    if quarter == 1:
        return f"{year - 1}.12.02..{year}.03.20"
    if quarter == 2:
        return f"{year}.03.21..{year}.06.24"
    if quarter == 3:
        return f"{year}.06.25..{year}.09.12"
    if quarter == 4:
        return f"{year}.09.13..{year}.12.01"


def create_tile_bboxes(
    metadata,
    tile_size_deg,
    latitude_col="lat",
    longitude_col="lon",
) -> list:
    """
    Create a grid of square tiles that cover the extent of the metadata.
    
    This function divides the geographic extent of the metadata into square tiles
    of the specified size. Each tile is defined by its bottom-left corner and dimensions.

    Args:
        metadata (pd.DataFrame): DataFrame containing species observation metadata
        tile_size_deg (tuple): Size of each tile in degrees (width, height)
        latitude_col (str, optional): Name of the latitude column. Defaults to "lat".
        longitude_col (str, optional): Name of the longitude column. Defaults to "lon".

    Returns:
        list: List of tile coordinates, each as [left, bottom, width, height] in degrees
    """
    min_latitude, max_latitude = (
        metadata[latitude_col].min(),
        metadata[latitude_col].max(),
    )
    min_longitude, max_longitude = (
        metadata[longitude_col].min(),
        metadata[longitude_col].max(),
    )
    latitude_step, longitude_step = tile_size_deg

    latitude_points = np.arange(min_latitude, max_latitude, latitude_step)
    longitude_points = np.arange(min_longitude, max_longitude, longitude_step)

    if len(latitude_points) < 1 or len(longitude_points) < 1:
        return []

    tile_bboxes = []
    for left in latitude_points:
        for bottom in longitude_points:
            tile_bboxes.append((left, bottom, latitude_step, longitude_step))

    return tile_bboxes


def get_metadata_in_tile(
    metadata, tile_bbox, latitude_col="lat", longitude_col="lon", padding=0.25
):
    """
    Filter metadata to include only observations within a tile's bounds plus padding.
    
    This function selects all observations that fall within the specified tile's
    boundaries, with an additional padding area around the edges.

    Args:
        metadata (pd.DataFrame): DataFrame containing species observation metadata
        tile_bbox (tuple): Tile coordinates [left, bottom, width, height] in degrees
        latitude_col (str, optional): Name of the latitude column. Defaults to "lat".
        longitude_col (str, optional): Name of the longitude column. Defaults to "lon".
        padding (float, optional): Additional padding around tile in degrees. Defaults to 0.25.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only observations within the tile bounds
    """
    left, bottom, width, height = tile_bbox

    df_in_tile = metadata[
        (left - padding <= metadata[latitude_col])
        & (metadata[latitude_col] <= (left + width + padding))
        & (bottom - padding <= metadata[longitude_col])
        & (metadata[longitude_col] <= (bottom + height + padding))
    ]

    return df_in_tile


def search_tile(
    position_df: pd.DataFrame,
    tile_bbox: tuple,
    raster_path: str,
    tile_image_output_dir: str = None,
    lat_column: str = "lat",
    lon_column: str = "lon",
) -> dict[int, Any]:
    """
    Extract data from a raster file for all observations within a tile.
    
    This function:
    1. Filters observations to those within the tile bounds
    2. Loads the raster data for the tile
    3. Extracts values for each observation
    4. Optionally saves image patches for each observation

    Args:
        position_df (pd.DataFrame): DataFrame containing observation locations
        tile_bbox (tuple): Tile coordinates [left, bottom, width, height] in degrees
        raster_path (str): Path to the raster file
        tile_image_output_dir (str, optional): Directory to save image patches. Defaults to None.
        lat_column (str, optional): Name of the latitude column. Defaults to "lat".
        lon_column (str, optional): Name of the longitude column. Defaults to "lon".

    Returns:
        dict[int, Any]: Dictionary mapping observation indices to their extracted values
    """
    # try:
    df_in_tile = get_metadata_in_tile(position_df, tile_bbox, lat_column, lon_column)
    if len(df_in_tile) == 0:
        return {}

    extractor = DataCubeExtractor(
        tile_bbox, band_index=1, lat_column=lat_column, lon_column=lon_column
    )
    # extractor = ImageDataCubeExtractor(tile_bbox, band_index=1, image_patch_size=64)

    has_data = extractor.load_raster(raster_path)

    if not has_data:
        print(f"No data in tile {tile_bbox}")
        return {}

    extracted_values_indexed = {}
    for index, row in df_in_tile.iterrows():
        extracted_values_indexed[index] = extractor[row]
        if tile_image_output_dir:
            extractor.save_patch_image(
                item=row, tile_image_output_dir=tile_image_output_dir
            )

    return {k: v for k, v in extracted_values_indexed.items() if v is not None}
