# -*- coding: utf-8 -*-
""" Down sampling module which can be used to downsample images by interpolation,
include sensor artifacts such as fillfactor and pixel location inaccuracies.

This module is directly based on the conference paper:

>>> Jean-José Orteu, Dorian Garcia, Laurent Robert, Florian Bugarin. A speckle-texture image generator.
    Slangen, P. Speckle06 - Speckles from grains to flowers, Sep 2006, Nimes, France. Extrait de: Speckle06
    - Speckles from grains to flowers - Proceedings of Spie / sous la dir. de P. SLANGEN et C. CERRUTTI.
    - ISBN 0-8194-6426-0, 6341, 2006, <10.1117/12.695280>. <hal-01644899>


"""

import numpy as np
from scipy import ndimage


def coord_subpos(coord, fillfactor, n, i, sigma):
    """
    Super sampled coordinates

    This function calculates the N sample coordinates with a given fillcator and in addition:
    - Inaccuracies in pixel position can be introduced by defining the standard deviation of a normal distribution
      which describes the pixel offset distribution.

    Parameters
    ----------
    coord : NxN array
        A coordinate array with the center coordinates about which the spread is calculated
    fill_factor : Float
        The area fraction covered by a pixel
    n : Int
        The number of super sampling points
    i : NxN array
        Index array for the super sampling points. Has the shape (n x n)
    permutation_sigma : Float
        The standard deviation of the normal distribution describing the pixel offset

    Returns
    -------
    Super sampled coordinates: N x N x n x n array


    Examples
    --------
    The following example bins the image in 4 x 4 bins

        >>> from muDIC import vlab
        >>> import numpy as np
        >>> pixel_centers = np.ones((1,1))
        >>> n=5
        >>> indices = np.arange(n)[:,np.newaxis] * np.ones((n,n))
        >>> vlab.downsampler.coord_subpos(pixel_centers, fillfactor=1.0, n, indices, sigma=0.):
    array([[[[0.6, 0.6, 0.6, 0.6, 0.6],
         [0.8, 0.8, 0.8, 0.8, 0.8],
         [1. , 1. , 1. , 1. , 1. ],
         [1.2, 1.2, 1.2, 1.2, 1.2],
         [1.4, 1.4, 1.4, 1.4, 1.4]]]])


    """
    distribution_mean = 0.
    perturbation = (np.sqrt(fillfactor) / n) * np.random.normal(distribution_mean, sigma)
    return coord[:, :, np.newaxis, np.newaxis] + np.sqrt(fillfactor) * (
            2. * i[np.newaxis, np.newaxis, :, :] + 1. - n) / (
                   2. * n) + perturbation


def chess_board(n):
    image_width = 2
    image_value = 2.0

    frame_with = 1
    frame_value = 1.0

    image = np.ones((image_width, image_width)) * image_value
    framed_image = np.pad(image, frame_with, mode="constant", constant_values=frame_value)

    tile = np.zeros((8, 8))
    tile[:4, :4] = framed_image
    tile[4:, :4] = framed_image * -1
    tile[:4, 4:] = framed_image * -1
    tile[4:, 4:] = framed_image

    tile_assembly = np.tile(tile, (n, n)).astype(np.float)

    return tile_assembly


class Downsampler(object):
    def __init__(self, image_shape, factor=4, fill=1.0, pixel_offset_stddev=0.0):
        """
        Downsampler for gray-scale images.

        This function downsamples a NxN image by an integer value with the following addition:
        - A fillfactor between 0 and 1 can be given in order to mimic a image sensors fillfactor
        - Inaccuracies in pixel position can be introduced by defining the standard deviation of a normal distribution
          which describes the pixel offset distribution.

        Parameters
        ----------
        image : NxN array
            The image which will be downsampled
        down_sampling_factor : Integer
            The downsampling factor
        fill_factor : Float
            The area fraction covered by a pixel
        permutation_sigma : Float
            The standard deviation of the normal distribution describing the pixel offset

        Returns
        -------
        downsampled_image: NxN array


        Examples
        --------
        The following example bins the image in 4 x 4 bins

        >>> image = np.ones((20,20))
            downsampled_image = downsample_image(image,4,fill_factor=1.0,permutation_sigma=0.0)
            downsampled_image.shape
        (5,5)


        """

        self.image_shape = image_shape
        self.factor = factor
        self.fill = fill
        self.pixel_offset_stddev = pixel_offset_stddev

        self.dim_n, self.dim_m = self.image_shape
        if self.dim_m != self.dim_n:
            raise ValueError("Only N x N images are supported, %i x %i was given" % (self.dim_m, self.dim_n))

        if divmod(self.dim_m, self.factor)[1] != 0:
            raise ValueError("Shape of image and downsampling factor does not match")

        if type(self.factor) is not int:
            raise TypeError("Down sampling factor has to be an integer")

        if self.fill > 1. or self.fill < 0.:
            raise ValueError("Fillfactor has to be between 1 and 0")

        self.coordinates = self.__generate_coordinates__()

    def __generate_coordinates__(self):

        n_subpoints = np.floor(self.factor * np.sqrt(self.fill)).astype(np.int)

        num_segments_n = np.float(self.dim_n) / self.factor
        num_segments_m = np.float(self.dim_m) / self.factor

        tile_i = np.arange(num_segments_n, dtype=np.float)
        tile_j = np.arange(num_segments_m, dtype=np.float)

        i, j = np.meshgrid(tile_i, tile_j)

        # Pertubate the pixels
        i_shift = np.random.normal(loc=0.0, scale=self.pixel_offset_stddev, size=i.shape)
        j_shift = np.random.normal(loc=0.0, scale=self.pixel_offset_stddev, size=j.shape)

        i += i_shift
        j += j_shift

        ind_i, ind_j = np.meshgrid(np.arange(n_subpoints, dtype=np.float), np.arange(n_subpoints, dtype=np.float))

        subframe_i = (coord_subpos(i, self.fill, n_subpoints, ind_i,
                                   self.pixel_offset_stddev) + .5) * np.float(
            self.factor) - .5

        subframe_j = (coord_subpos(j, self.fill, n_subpoints, ind_j,
                                   self.pixel_offset_stddev) + .5) * np.float(
            self.factor) - 0.5

        coordinates = np.zeros(
            (2, np.int(num_segments_n), np.int(num_segments_m), n_subpoints, n_subpoints))

        coordinates[1, :, :, :, :] = subframe_i
        coordinates[0, :, :, :, :] = subframe_j
        return coordinates

    def __call__(self, image):
        if image.shape[0] != self.dim_m or image.shape[1] != self.dim_n:
            raise ValueError("Invalid shape")

        downsampled_image_bins = ndimage.map_coordinates(image, self.coordinates, order=3, prefilter=True, cval=np.nan)

        return np.mean(downsampled_image_bins, axis=(2, 3))
