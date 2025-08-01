# based on https://github.com/chamidullinr/CVAT-Utils/tree/dev

from typing import Tuple

import cv2
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageDraw


def rle_to_mask(points: list, image_height: int, image_width: int) -> np.ndarray:
    """Decode data compressed with CVAT-based Run-Length Encoding (RLE) and return image mask.

    Parameters
    ----------
    points
        List of points stored in shape with mask type returned by CVAT API.
    image_height
        Height of image.
    image_width
        Width of image.

    Returns
    -------
    mask
        A 2D NumPy array that represent image mask.
    """
    points = np.array(points).astype(int)
    rle = points[:-4]
    left, top, right, bottom = points[-4:]
    width = right - left + 1
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    value, offset = 0, 0
    for rle_count in rle:
        while rle_count > 0:
            x, y = offset % width, offset // width
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = abs(value - 1)
    return mask


def mask_to_points(mask: np.ndarray) -> list:
    """Convert 2D mask into a 1D flattened list of points.

    Parameters
    ----------
    mask
        A 2D NumPy array that represent image mask.

    Returns
    -------
    points
        A 1D flattened list of points.
    """
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )[0]
    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception("Less then three point have been detected. Can not build a polygon.")
    return contours.reshape(-1).tolist()


def points_to_mask(image: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    """Converts polygon points to binary mask.

    Parameters
    ----------
    image
        Given image as np.array.
    polygon_points:
        Polygon in a format [x1, y1, x2, y2 ...]
    Returns
    -------
        Binary Mask as np.ndarray.
    """
    height, width, _ = image.shape
    x_coordinates = polygon_points[::2]
    y_coordinates = polygon_points[1::2]
    polygon = list(zip(x_coordinates, y_coordinates))

    image = Image.new("L", (int(width), int(height)), 0)
    ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)

    return np.array(image)