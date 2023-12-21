from torchvision import transforms
import numpy as np
import torch
from scipy import ndimage

import functools
from PIL import Image


class PairTransform:
    def __init__(self, dataset, type_transform="Shuffle2D"):
        self.type_transform = type_transform
        self.dataset = dataset
        if self.dataset == 'stl10':
            s = 1
            size = 96
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=[128, 128]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomApply([color_jitter], p=0.8),
                                                 transforms.RandomGrayscale(p=0.2),
                                                 GaussianBlur(kernel_size=int(0.1 * size)),
                                                 transforms.ToTensor()])
        elif self.dataset == "cifar10":
            print("cifar10")
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

        elif self.dataset == 'DRPDataset2D':
            custom_transformations = {
                "VerticalFlip": transforms.RandomVerticalFlip(p=0.5),
                "HorizontalFlip": transforms.RandomHorizontalFlip(p=0.5),
                "GaussianBlur": GaussianBlur(p=0.3, kernel_size=3, sigma=(0.1, 2)),
                "Shuffle2D": Shuffle2D(nb_patches=4, p=0.7),
                "DistanceTransform": Distance_Transform_EDT(p=1),
                "Cutout": Cutout(p=0.5, n_holes=5, length=30),
                "LocalShuffling": LocalShuffling(p=1, window=(2, 2)),
            }

            if isinstance(self.type_transform, str):
                self.type_transform = self.type_transform.split(", ")
                # self.type_transform = [self.type_transform]

            selected_transforms = []
            for transform_name in self.type_transform:
                print(transform_name)
                selected_transforms.append(custom_transformations[transform_name])

            self.transform = transforms.Compose([
                *selected_transforms,
                transforms.ToTensor()])

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            ])

    def __call__(self, sample):
        if self.type_transform == 'Combine':
            xi = self.transform(sample)
            xj = self.transform(sample)
        else:
            xi = self.transform(sample)
            xj = self.transform(sample)
        return [xi, xj]


class GaussianBlur:
    def __init__(self, p=1, kernel_size=3, sigma=(0.1, 2)):
        self.p = p
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image):
        if torch.rand(1) < self.p:
            return self.transform(image)
        else:
            return image


# class Shuffle2D:
#     def __init__(self, nb_patches, p=1):
#         """
#         Split an image into 'nb_parts' number of parts and shuffle them randomly.
#
#         Parameters:
#         nb_parts (int): Number of parts to split the image.
#         p (float): Probability of applying the transformation (default: 1).
#         """
#         self.nb_parts = nb_patches
#         self.p = p
#
#     def __call__(self, image):
#         """
#         Apply the transformation to the input image.
#
#         Parameters:
#         img (PIL.Image.Image): The input image.
#
#         Returns:
#         PIL.Image.Image: The shuffled image.
#         """
#         if torch.rand(1) < self.p:
#             width, height = image.size
#             patch_width = width // self.nb_parts
#             patch_height = height // self.nb_parts
#
#             patches = []
#             for i in range(self.nb_parts):
#                 for j in range(self.nb_parts):
#                     patch = image.crop((i * patch_width, j*patch_height, (i + 1) * patch_width, (j + 1) * patch_height))
#                     patches.append(patch)
#
#             np.random.shuffle(patches)
#
#             shuffled_image = Image.new('L', (width, height))
#             current_patch = 0
#
#             for i in range(self.nb_parts):
#                 for j in range(self.nb_parts):
#                     shuffled_image.paste(patches[current_patch], (i * patch_width, j * patch_height))
#                     current_patch += 1
#
#             return shuffled_image
#         else:
#             return image
class Shuffle2D():
    def __init__(self, nb_patches, p):

        self.nb_patches = nb_patches
        self.p = p

    def __call__(self, image):

        width, height = image.size

        if torch.rand(1) < self.p:
            image = np.array(image)
            image_shuffled = np.zeros((width, height))
            patch = int(width/self.nb_patches)

            list_patch = []
            for x in range(0, patch*self.nb_patches, patch):
                for y in range(0, patch*self.nb_patches, patch):

                    list_patch.append(image[x:x+patch, y:y+patch])

            np.random.shuffle(list_patch)

            h = 0
            for i in range(0, patch*self.nb_patches, patch):
                for j in range(0, patch*self.nb_patches, patch):
                    for l in range(patch):
                        for k in range(patch):

                            image_shuffled[i+l, j+k] = list_patch[h][l, k]

                    h += 1
            # image_shuffled = (image_shuffled - np.min(image_shuffled))/(np.max(image_shuffled)-np.min(image_shuffled))
            return Image.fromarray(image_shuffled)
        else:
            # return (image - np.min(image))/(np.max(image)-np.min(image))
            return image

class Cutout:
    def __init__(self, n_holes, length, p=1):
        """
            Randomly mask out one or more patches from an image.

            Parameters:
                n_holes (int): Number of patches to cut out of each image.
                length (int or tuple): The length (in pixels) of each square patch. If an integer is provided,
                                           both dimensions will have the same length.
                p (float): Probability of applying the transformation (default: 1).
        """
        self.n_holes = n_holes
        self.length = length
        self.p = p
        if type(self.length) is int:
            self.length = (self.length, self.length)

    def __call__(self, image):
        """
                Apply the transformation to the input image.

                Parameters:
                    image (PIL.Image.Image): The input image.

                Returns:
                    PIL.Image.Image: The image with randomly masked out patches.
                """
        if torch.rand(1) < self.p:
            width, height = image.size
            image = np.array(image)

            for i in range(self.n_holes):
                offset = generate_offset(fullshape=(width, height), subshape=self.length)
                image[offset[0]:offset[0] + self.length[0], offset[1]:offset[1] + self.length[1]] = 0

            return Image.fromarray(image)
        else:
            return image


def generate_offset(fullshape, subshape):
    """
    Generate random offsets to define a patch position within an image.

    Parameters:
        fullshape (tuple): The dimensions (width, height) of the full image.
        subshape (tuple): The dimensions (width, height) of the patch to be masked.

    Returns:
        tuple: The random offset position (x, y).
    """
    max_x = fullshape[0] - subshape[0]
    max_y = fullshape[1] - subshape[1]
    offset_x = np.random.randint(0, max_x + 1)
    offset_y = np.random.randint(0, max_y + 1)
    return offset_x, offset_y


def is_valid_offset(subshape, offset, fullshape):
    return functools.reduce(
        lambda x, y: x and y,
        map(lambda x, y, z: 0 <= (x + y) <= z, subshape, offset, fullshape),
    )


class LocalShuffling:
    def __init__(self, p: float, window: tuple):
        self.p = p
        self.window = window

    @staticmethod
    def _check_argument(window: tuple) -> None:
        if len(window) != 2 or not all(isinstance(x, int) for x in window):
            raise ValueError(
                'window must be 2-dimensional containing only integer numbers'
            )
        elif min(window) <= 0:
            raise ValueError('size window values must be greater than 0')

    def __call__(self, image):
        if torch.rand(1) < self.p:
            shuffled = image.copy()
            self.x, self.y = shuffled.size
            image, shuffled = np.asarray(image), np.asarray(shuffled)

            x_list = list(range(self.window[0], self.x + 1, self.window[0]))
            y_list = list(range(self.window[1], self.y + 1, self.window[1]))

            for x_n, x_piece in enumerate(x_list):
                for y_n, y_piece in enumerate(y_list):
                    patch = image[y_n * self.window[0]:y_piece, x_n * self.window[1]:x_piece]
                    np.random.shuffle(patch)
                    shuffled[y_n * self.window[0]:y_piece, x_n * self.window[1]:x_piece] = patch[:, :]
            return shuffled
        else:
            return image


class Distance_Transform_EDT:
    """
    Distance transform to enhance the amount of accessible information in input data
    p: probability of the transform occurring
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        image = np.array(image)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        img_resul = np.zeros(image.shape)
        if torch.rand(1) < self.p:

            for threshold in range(255):
                img_mask1 = 1.0 * (image >= threshold / 255.0) #create a mask binary for each grayscale intensity
                img_mask2 = 1.0 * (image < threshold / 255.0) #create a inverted mask binary for each grayscale intensity
                if threshold == 0:
                    img_edt1 = ndimage.distance_transform_edt(img_mask1)
                    img_edt2 = ndimage.distance_transform_edt(img_mask2)
                else:
                    img_edt1 += ndimage.distance_transform_edt(img_mask1) # Distance transform starting from lowest to highest grayscale intensities
                    img_edt2 += ndimage.distance_transform_edt(img_mask2)

            img_resul = - img_edt1 + img_edt2
            img_resul = (img_resul - np.min(img_resul)) / (np.max(img_resul) - np.min(img_resul))

        return (img_resul + image - np.min(img_resul + image)) / (np.max(img_resul + image) - np.min(img_resul + image))

