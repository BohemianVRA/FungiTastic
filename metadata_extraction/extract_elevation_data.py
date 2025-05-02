"""
Extract elevation data from ASTER elevation raster files for fungi observations.

This script processes a CSV file containing fungi observation metadata and extracts
elevation data for each observation from an ASTER elevation raster file. The process:

1. Loads observation metadata from a CSV file
2. Divides the geographic extent into tiles for parallel processing
3. For each tile:
   - Extracts elevation values for observations within the tile
   - Optionally saves image patches around each observation
4. Combines results and saves to a new CSV file

The script uses parallel processing to speed up extraction across multiple tiles.

Required environment variables:
    None

Required files:
    - Input metadata CSV with columns: observationID, latitude, longitude
    - ASTER elevation raster file (.tif)

Example usage:
    python extract_elevation_data.py
"""

import multiprocessing
import os
import os.path as osp
import threading
import time
from collections import ChainMap
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from datacube_extractor import DataCubeExtractor, ImageDataCubeExtractor
from joblib import Parallel, delayed
from PIL import Image
from tqdm.auto import tqdm

from utils import create_tile_bboxes, search_tile


if __name__ == "__main__":
    # Configuration
    n_jobs = multiprocessing.cpu_count()  # Number of parallel processes

    # Input/output paths
    subset = "train"
    source = ("-FewShot")
    metadata_path = (
        f"../metadata/FungiTastic{source}/FungiTastic{source}-{subset}-metadata.csv"
    )
    raster_path = "/Users/lukaspicek/Downloads/rasters/EnviromentalRasters/Elevation/ASTER_Elevation.tif"
    output_dir = "../metadata/"

    # Processing parameters
    create_images = False  # Whether to save image patches
    tile_size_deg = (4, 4)  # Size of processing tiles in degrees
    latitude_col_name = "latitude"  # Name of latitude column in metadata
    longitude_col_name = "longitude"  # Name of longitude column in metadata
    unique_id = "observationID"  # Column containing unique observation IDs
    raster = "elevation"  # Name for the extracted data column

    output_file_name = f"../metadata/FungiTastic{source}/FungiTastic{source}-{subset}-metadata-{raster}.csv"

    # Load and preprocess metadata
    print("Loading metadata...")
    metadata = pd.read_csv(
        metadata_path,
        delimiter=",",
        low_memory=False,
    )

    # Remove duplicate observations and prepare output dataframe
    metadata = metadata.drop_duplicates(unique_id).reset_index(drop=True)
    out_metadata = metadata[[unique_id]].copy(deep=True)

    # Create processing tiles
    print("Creating processing tiles...")
    tile_bboxes = create_tile_bboxes(
        metadata,
        tile_size_deg,
        latitude_col_name,
        longitude_col_name,
    )

    # Verify raster file exists
    if not os.path.isfile(raster_path):
        print(f"Error: Raster file not found at {raster_path}")
        exit(1)

    # Extract elevation data in parallel
    print(f"Extracting elevation data using {n_jobs} processes...")
    start_time = time.time()
    extracted_tiles = Parallel(n_jobs=n_jobs)(
        delayed(search_tile)(
            metadata,
            tile_bbox,
            raster_path,
            lat_column=latitude_col_name,
            lon_column=longitude_col_name,
        )
        for tile_bbox in tqdm(tile_bboxes, total=len(tile_bboxes))
    )

    # Combine results from all tiles
    print("Combining results...")
    extracted_values_indexed = dict(ChainMap(*extracted_tiles))

    # Add elevation data to output metadata
    out_metadata[raster] = out_metadata.index.map(extracted_values_indexed)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # Save results
    print(f"Saving results to {output_file_name}...")
    out_metadata.to_csv(f"{output_dir}/{output_file_name}.csv", index=False)
    print("Done!")
