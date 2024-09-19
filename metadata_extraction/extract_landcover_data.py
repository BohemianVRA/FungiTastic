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
    n_jobs = multiprocessing.cpu_count()

    subset = "val"
    source = ("")
    metadata_path = (
        f"../metadata/FungiTastic{source}/FungiTastic{source}-{subset}-metadata.csv"
    )

    raster_path = "/Users/lukaspicek/Downloads/rasters/EnviromentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif"
    output_dir = "../metadata/"

    create_images = False
    tile_size_deg = (4, 4)
    latitude_col_name = "latitude"
    longitude_col_name = "longitude"
    unique_id = "observationID"
    raster = "landcover"

    output_file_name = f"../metadata/FungiTastic{source}/FungiTastic{source}-{subset}-metadata-{raster}.csv"

    metadata = pd.read_csv(
        metadata_path,
        delimiter=",",
        low_memory=False,
    )

    header = []
    metadata = metadata.drop_duplicates(unique_id).reset_index(drop=True)
    out_metadata = metadata[[unique_id]].copy(deep=True)

    time_stamp = 0

    tile_bboxes = create_tile_bboxes(
        metadata,
        tile_size_deg,
        latitude_col_name,
        longitude_col_name,
    )

    if not os.path.isfile(raster_path):
        print(f"Raster not found. Skipping!")

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

    if not os.path.isfile(raster_path):
        print(f"Raster not found. Skipping!")
        exit()

    extracted_values_indexed = dict(ChainMap(*extracted_tiles))

    out_metadata[raster] = out_metadata.index.map(extracted_values_indexed)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    out_metadata.to_csv(f"{output_dir}/{output_file_name}", index=False)
    print(f"Done!")


