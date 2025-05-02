import datetime
import os
import threading
import time

import geopandas
import numpy as np
import tqdm
from ecodatacube_extractor import EcodatacubeExtractor
from PIL import Image


def extract_ecodatacube(
    df,
    data_path,
    output_path,
    bands,
    cell_size=100000,
    margin=1000,
    patch_size=128,
    match_year=False,
    quality=90,
    gamma=2.5,
    convert_uint8=True,
    patchId_name="surveyId",
    lat_name="lat",
    lon_name="lon",
    year_name="year",
):
    # initialize timer
    start_time = time.time()

    global extractors
    global next_extractors

    # function that instantiate an extractor (download a raster tile for extraction of patch)
    def prep_extractor(raster_url, num_band, grid_cell, cell_size, patch_size):
        global next_extractors
        next_extractors[num_band] = EcodatacubeExtractor(
            raster_url,
            grid_cell,
            cell_size,
            patch_size,
            margin=margin,
            gamma=gamma,
            convert_uint8=convert_uint8,
        )

    def split_coords(point):
        return point.x, point.y

    df.rename(
        columns={
            patchId_name: "PlotID",
            lon_name: "lon",
            lat_name: "lat",
            year_name: "year",
        },
        inplace=True,
    )
    if match_year:
        df.rename(columns={year_name: "year"}, inplace=True)
    df = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    df = df.to_crs(epsg=3035)
    df["x_EPSG3035"], df["y_EPSG3035"] = zip(*df["geometry"].map(split_coords))

    # keep only one row for each unique PlotID
    df = df.drop_duplicates(subset="PlotID", keep="first", ignore_index=True)

    # Convert the x and y coordinates to integer values representing the grid cells
    df["grid_x"] = (df["x_EPSG3035"] // cell_size).astype(int)
    df["grid_y"] = (df["y_EPSG3035"] // cell_size).astype(int)

    # Create a column with the cell indices as a tuple
    df["grid_cell"] = list(zip(df["grid_y"], df["grid_x"]))
    grid_cells = df["grid_cell"].unique()
    if match_year:
        years = df["year"].unique()
    else:
        years = [2021]

    for year in years:
        if match_year:
            df_year = df.loc[df["year"] == year]
        else:
            df_year = df
        raster_paths = [
            data_path
            + "lcv_"
            + str(band)
            + "_sentinel.s2l2a_p50_30m_0..0cm_"
            + str(year)
            + ".06.25.."
            + str(year)
            + ".09.12_eumap_epsg3035_v1.0.tif"
            for band in bands
        ]
        print("Extraction of year " + str(year) + "!")
        try:
            # Prepare the first tile
            next_extractors = [
                EcodatacubeExtractor(
                    raster_path,
                    grid_cells[0],
                    cell_size,
                    patch_size,
                    margin=margin,
                    gamma=gamma,
                    convert_uint8=convert_uint8,
                )
                for raster_path in raster_paths
            ]

            # loop on grid_cells (= tiles) to extract patches on each tiles
            for i, grid_cell in tqdm.tqdm(enumerate(grid_cells), total=len(grid_cells)):
                # next_extractor become current extractor
                extractors = next_extractors
                next_extractors = [None for j in range(len(next_extractors))]

                # create new threads to download in parallel next tiles and instantiate the next
                # extractors
                if i < (len(grid_cells) - 1):
                    threads_next = [
                        threading.Thread(
                            target=prep_extractor,
                            args=(
                                raster_paths[j],
                                j,
                                grid_cells[i + 1],
                                cell_size,
                                patch_size,
                            ),
                        )
                        for j in range(len(raster_paths))
                    ]
                    # start the new threads
                    for thread in threads_next:
                        thread.start()

                # get all occurrences that are within the tile
                df_cell = df_year.loc[df_year["grid_cell"] == grid_cell]

                # iterate through each row in the dataframe
                for index, row in df_cell.iterrows():
                    try:
                        PlotID = int(row["PlotID"])
                        lat = row["lat"]
                        lon = row["lon"]

                        # construct the outup file path of the patch as './CD/AB/XXXXABCD.jpeg'
                        path = output_path
                        for d in (str(PlotID)[-2:], str(PlotID)[-4:-2]):
                            path = os.path.join(path, d)
                            if not os.path.exists(path):
                                os.makedirs(path)

                        patch_layers = []
                        # call the patch extractor to get the numpy patch
                        if len(extractors) > 1:
                            for extractor in extractors:
                                patch_layers.append(extractor[(lat, lon)][np.newaxis])
                            patch = np.concatenate(patch_layers, axis=0).transpose(
                                1, 2, 0
                            )
                            image_name = f"{PlotID}.jpeg"
                            image = Image.fromarray(patch)
                        else:
                            extractor = extractors[0]
                            patch = extractor[(lat, lon)]
                            image_name = f"{PlotID}.jpeg"
                            image = Image.fromarray(patch)
                            image = image.convert("L")
                        # save the patch as a jpeg with a specific compression rate
                        image.save(
                            os.path.join(path, image_name), "JPEG", quality=quality
                        )
                    except Exception as e:
                        # if an error occurs for a patch, save the id of the occurrence
                        with open(output_path + "errors.txt", "a") as f:
                            f.write(str(PlotID) + ";" + str(e) + "\n")
                # wait for the next extractor to be ready and get it
                if i < (len(grid_cells) - 1):
                    for thread in threads_next:
                        thread.join()
        except Exception as e:
            print(e)
            print("error during extraction of occurrences from year " + str(year) + "!")

    print("finished !")
    # print total execution time
    t = str(datetime.timedelta(seconds=(time.time() - start_time)))
    print("--- " + t + " ---")
