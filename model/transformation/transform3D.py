import numpy as np
from scipy import ndimage
from simclr_stl.simclr.data.load_cube import is_valid_offset


def convert2gray(arr, threshold=None):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    if arr.max() > 1.0:
        arr = (arr - np.min(arr))*255.0/(np.max(arr)-np.min(arr))
    hist, _ = np.histogram(arr, bins=256)
    hist = gaussian_filter1d(hist, 5)
    peaks, _ = find_peaks(hist, height=0)

    threshold = (peaks[0]+peaks[1])/2 if threshold is None else threshold
    segmented_arr = np.where(arr < threshold, 0, 255)
    return segmented_arr


def distance_transform(arr):
    segmented_arr = convert2gray(arr, threshold=None)
    segmented_arr = np.where(segmented_arr == 255, 0, 255)
    img_edt = ndimage.distance_transform_edt(segmented_arr)

    return img_edt


def local_pixel_shuffling(arr, size_window=4):
    img_shuffled = arr.copy()
    for x in range(0, img_shuffled.shape[0], size_window):
        for y in range(0, img_shuffled.shape[1], size_window):
            for z in range(0, img_shuffled.shape[2], size_window):
                window = img_shuffled[x:x+size_window, y:y+size_window, z:z+size_window]
                np.random.shuffle(window)
                img_shuffled[x:x+size_window, y:y+size_window, z:z+size_window] = window[:, :, :]
    return img_shuffled


def non_linear_intensity(arr, number_bins=256):
    image_histogram, bins = np.histogram(arr.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(arr.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(arr.shape)

# def Shuffled_Path(arr, nb_patches=4):
#     from copy import copy
#     img_shuffled = []
#     for x in range(0,arr.shape[0],int(arr.shape[0]/nb_patches)):
#         for y in range(0,arr.shape[1],int(arr.shape[1]/nb_patches)):
#             for z in range(0,arr.shape[2],int(arr.shape[2]/nb_patches)):
#                 window = arr[x:x+int(arr.shape[0]/nb_patches),y:y+int(arr.shape[1]/nb_patches),z:z+int(arr.shape[2]/nb_patches)]
#                 img_shuffled.append(window)
#     np.random.shuffle(img_shuffled)
#     img_shuffled = np.array(img_shuffled).reshape(arr.shape)
#     # img_shuffled[x:x+size_window,y:y+size_window,z:z+size_window] = window[:,:,:]
#     patches = patchify(arr, (int(arr.shape[0]/nb_patches), int(arr.shape[0]/nb_patches), int(arr.shape[0]/nb_patches)), step=int(arr.shape[0]/nb_patches))
#     return img_shuffled


def random_erasing(arr, nb_eraser=3):
    import random
    eraser = [20, 40, 80, 60]
    img_erased = arr.copy()
    offset = None
    for i in range(nb_eraser):
        size_eraser = random.choice(eraser)
        offset = (random.randint(1, arr.shape[0]), random.randint(1, arr.shape[1]), random.randint(1, arr.shape[2]))
        cube_shape = arr.shape
        subshape = (size_eraser, size_eraser, size_eraser)
        if is_valid_offset(subshape, offset, cube_shape):
            img_erased[offset[0]:offset[0]+subshape[0],
                       offset[1]:offset[1]+subshape[1],
                       offset[2]:offset[2]+subshape[2]] = 0.0

    return img_erased, offset
