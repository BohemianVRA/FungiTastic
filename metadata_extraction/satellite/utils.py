from typing import Any

import numpy as np
import pandas as pd
from datacube_extractor import DataCubeExtractor, ImageDataCubeExtractor


def get_quarter_dates(year: str, quarter: str) -> str:
    """
    Returns quarter formatted date strings for ecodatacude data extraction.
    :param year: year
    :param quarter: quarter
    :return: string with reference to given quarter and string
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
    Creates squared tiles of a given size that fits within given extend.
    :param metadata: pd.dataframe with metadata about the species observations
    :param tile_size_deg: size of a tile in degrees
    :param latitude_col: name of a column in the metadata file containing latitude
    :param longitude_col: name of a column in the metadata file containing longitude
    :return: list wit tiles coordinates in degrees [[left, bottom, width, height],...]
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
    Returns all observations for a given tile + a little bit around based on padding.
    :param metadata: pd.dataframe with species observations  metadata
    :param tile_bbox: tile coordinates in degrees as list with [left, bottom, width, height]
    :param latitude_col: name of a column in the metadata file containing latitude
    :param longitude_col: name of a column in the metadata file containing longitude
    :param padding:
    :return:
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
