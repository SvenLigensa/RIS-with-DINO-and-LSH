# Copyright (c) 2022 Sven Ligensa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os
import rasterio
from rasterio.windows import Window
from rasterio.plot import show


def get_patch(id: int, w: int = 512, h: int = 512):
    """Returns the pixel values of the given patch."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        num_cols = int(src.width/w)
        row_px = int((id % num_cols) * w)
        col_px = int((id // num_cols) * h)
        arr = src.read(window=Window(row_px, col_px, w, h))
    return arr


def display_patch(id: int, w: int = 512, h: int = 512):
    """Displays the given patch."""
    show(get_patch(id, w, h))


def print_dataset_information(dataset):
    """Prints information about the given GeoTIFF dataset."""
    print("Info about the dataset\n"
        f"Name: {dataset.name}\n"
        f"Meta: {dataset.meta}\n"
        f"Bounds: {dataset.bounds}\n"
        f"Number of Bands: {dataset.count}\n"
        f"Width: {dataset.width}\n"
        f"Height: {dataset.height}\n"
        f"Coordinate reference system (CRS): {dataset.crs}\n"
        f"Spacial position of upper left corner: {dataset.transform * (0,0)}\n"
        f"Spacial position of lower right corner: {dataset.transform * ((dataset.width, dataset.height))}")
    print({i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)})  # Datatypes of all 3 bands


def get_info_patches(row: int,
                     w: int = 512,
                     h: int = 512,
                     ratio_nonzero_allowed: float = 0.1):
    """Returns an array of indices of patches containing less than ratio_nonzero_allowed zeros for a given row."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        num_cols = get_cols(w, h)
        patch = row*num_cols
        # Indices of the patches containing enough information
        info_indices = np.array([])
        current_row = 0    # Row counter
        for col in range(0, num_cols):
            # Read the patch by accessing the source at the specified window
            win = Window(col * w, row * h, w, h)
            arr = src.read(window=win)
            # Check, if patch contains enough information
            if ((np.sum(arr == 0) + np.sum(arr == 255)) < ratio_nonzero_allowed * np.prod(arr.shape)):
                info_indices = np.append(info_indices, int(patch))
            patch += 1
    return info_indices


def get_rows(w: int = 512, h: int = 512):
    """Returns the total number of rows of patches."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        num_rows = int(src.height/h)
    return num_rows


def get_cols(w: int = 512, h: int = 512):
    """Returns the total number of columns of patches."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        num_cols = int(src.width/w)
    return num_cols


def get_row(idx: int, w: int = 512, h: int = 512):
    """Returns the row of the patch with the given index in the source file."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        row_number = id // get_cols()
    return row_number


def get_col(idx: int, w: int = 512, h: int = 512):
    """Returns the column of the patch with the given index in the source file."""
    with rasterio.open(os.getenv('IMG_PATH')) as src:
        col_number = id % get_cols()
    return row_number
