import os
from ast import literal_eval
import numpy as np
import functools


def is_valid_offset(subshape, offset, fullshape):
    """
    Check if the required offset and subshape are within data shape
    :return:
    """
    return functools.reduce(
        lambda x, y: x and y,
        map(lambda x, y, z: 0 <= (x + y) <= z, subshape, offset, fullshape),
    )


def load_cube(path, idx_cube, modality, offset=None, subshape=None):
    path = path + str(idx_cube) + '/'
    roi_file = next(filter(
                lambda path: str(modality).lower() in path.lower()
                and path.lower().endswith(".dat"),
                os.listdir(path),
            ), None)
    assert (
        roi_file is not None
    ), f"memmap data not found for modality {str(modality).lower()} and root {path}"
    cube_params = roi_file.split("-")
    cube_shape = literal_eval(cube_params[2].split(".")[0])
    # load data as memmap array
    memmap_array = np.memmap(
        os.path.join(path, roi_file),
        dtype=np.dtype(cube_params[1]),
        mode="r",
        shape=cube_shape,
    )
    if offset is None:
        offset = (0, 0, 0)

    if subshape is None:
        subshape = cube_shape

    if is_valid_offset(subshape, offset, cube_shape):
        memmap_array = memmap_array[
            offset[0]: subshape[0] + offset[0],
            offset[1]: subshape[1] + offset[1],
            offset[2]: subshape[2] + offset[2],
        ]
    else:
        raise Exception(
            f"Subshape {subshape} and offset {offset} are not valid for shape {cube_shape}"
        )
    return memmap_array
