import random

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from noise import pnoise2
except ImportError as e:
    print(e)
    print("The package: noise, is not installed. Perlin speckle is not available")


    def pnoise2(*args, **kwargs):
        raise ImportError("The noise package is not installed")


def insert_circle(image, position, radius, allow_overlap):
    try:
        circle = np.zeros((radius * 2, radius * 2))
        xs, ys = np.meshgrid(np.arange(2 * radius), np.arange(2 * radius))
        r = ((xs - radius) ** 2. + (ys - radius) ** 2.) ** 0.5
        circle[r < radius] = 1.

        if not allow_overlap:
            non_zeros = np.count_nonzero(
                image[position[0] - radius:position[0] + radius, position[1] - radius:position[1] + radius])
            if non_zeros > 0:
                raise IndexError("Overlapping dots are not allowed")

        image[position[0] - radius:position[0] + radius, position[1] - radius:position[1] + radius] += circle
    except Exception as e:
        pass

    return image


def dots_speckle(size=(1000, 1000), n_dots=5500, dot_radius_max=40, dot_radius_min=30, blur_sigma=2,
                 allow_overlap=False):
    """ Speckle made by dots

    Returns a speckle looking like dots from a circular marker


    Example
    -------
    Let us make a speckle image with a size of 1000 x 1000 pixels, with approx 5000 dots, the smallest being 20 pixels
    and the largest being 25 pixels, all without overlap. Lets blur it to "round" the dots a little.

    The following code generates such a speckle

    >>> import muDIC as dic
    >>> speckle = dic.dots_speckle((1000,1000),n_dots=5000,dot_radius_max=25,dot_radius_min=20,blur_sigma=2,allow_overlap=False)




    Note
    ----
    The dots are not overlapping, the number of dots specified is the number of attempts to fit a dot on the image.

    Parameters
    ----------

    size=(1000,1000), n_dots=5500, dot_radius_max = 40,dot_radius_min = 30, blur_sigma=2,allow_overlap=False

    size : tuple
        The image size as a tuple of integers
    n_dots : int
        The number of dots in the image. Note that this corresponds to the number of attempts if overlap is not allowed.
    dot_radius_max: float, int
        The largest radius of a dot in the image
    dot_radius_min: float, int
        The smallest radius of a dot in the image
    blur_sigma : float
        The standard deviation of the gaussian kernel used to create gradients in the speckle image
    allow_overlap : bool
        Allow for overlapping dots
    """

    size_x, size_y = size
    img = np.zeros((size_x, size_y))

    for i in range(n_dots):
        pos_x = np.int(random.random() * size_x)
        pos_y = np.int(random.random() * size_y)

        radius = np.int(random.random() * (dot_radius_max - dot_radius_min) + dot_radius_min)

        img = insert_circle(img, (pos_x, pos_y), radius=radius, allow_overlap=allow_overlap)
    filtered = gaussian_filter(img, blur_sigma)

    filtered_normalized = normalize_array_to_unity(filtered)

    return filtered_normalized * -1. + 1.


def harmonic_speckle(size=(1000, 1000), n_peaks_x=20):
    """ Speckle made by harmonic functions

    Returns a speckle looking like a bilateral wave pattern


    Example
    -------
    Let us make a speckle image with a size of 1000 x 1000 pixels, with 20 peaks along the first axis

    The following code generates such a speckle

    >>> import muDIC as dic
    >>> speckle = dic.harmonic_speckle((1000,1000),n_peaks_x=20)


    Parameters
    ----------

    size=(1000,1000), n_dots=5500, dot_radius_max = 40,dot_radius_min = 30, blur_sigma=2,allow_overlap=False

    size : tuple
        The image size as a tuple of integers
    n_peaks_x : int
        The number of peaks along the first axis of the image
    """
    size_x, size_y = size
    xs, ys = np.meshgrid(np.arange(size_x), np.arange(size_y))

    freq = np.pi * 2. * np.float(n_peaks_x) / size_x
    x_harm = np.sin(xs * freq)
    y_harm = np.sin(ys * freq)

    field = x_harm * y_harm

    return normalize_array_to_unity(field)


def normalize_array_to_unity(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def smooth_step(array, c):
    return 0.5 * (1. + np.tanh((array) / c ** 2.))


def perlin_noise_speckle(shape, multiplier=64., octaves=1):
    """ Perlin noise based speckle

    Returns a speckle made by using Perlin noise provided by the Noise package by Casey Duncan (caseman).


    Example
    -------
    Let us make an example perlin noise spackle with a size of 1000x1000 pixels using the defaults.

    The following code generates such a speckle

    >>> import muDIC as dic
    >>> speckle = dic.perlin_noise_speckle((1000,1000))

    Notes
    -------
    The speckle generator uses the "pnoise2" function of the noise library, so you can look at the docs for that library
    for further documentation.

    Parameters
    ----------

    shape : tuple
        The image size as a tuple of integers
    multiplier : float
        The frequency multiplier
    octaves: float, int
        The number of octaves used
    """

    freq = multiplier * octaves
    img = np.zeros(shape)
    n, m = shape

    for y in range(n):
        for x in range(m):
            img[x, y] = float(pnoise2(x / freq, y / freq, octaves) * (float(n) - 1.) + float(n))

    img = normalize_array_to_unity(img) * 2.
    img = smooth_step(img.astype(np.float), c=0.7)
    img = normalize_array_to_unity(img)

    return img


def rosta_speckle(size, dot_size=4, density=0.32, smoothness=2.0):
    """ Rosta speckle

     Returns a speckle made by the "Rosta" algorithm
     This algorithm is very pragmatic and makes a speckle looking like a "real" spray speckle without
     being based on any real physics.



     Example
     -------
     Let us make a speckle image with a size of 1000 x 1000 pixels, using the default values.

     The following code generates such a speckle

     >>> import muDIC as dic
     >>> speckle = rosta_speckle((1000,1000), dot_size=4, density=0.32, smoothness=2.0, layers=1)

    If you want a denser speckle, you can increase the number of layers like this:

     >>> speckle = rosta_speckle((1000,1000), dot_size=4, density=0.32, smoothness=2.0, layers=1)


     Parameters
     ----------

     size : tuple
         The image size as a tuple of integers
     dot_size : int
         The size of the dots [0,1]
     density: float, int
        How packeg with dots the speckle should be
     smoothness: float, int
        The degree of smoothing applied to the binary speckle
     """
    merge_sigma = dot_size * size[0] / 1000.
    blur_sigma = smoothness * size[0] / 1000.

    noise = np.random.randn(*size)
    noise_blurred = gaussian_filter(noise, sigma=merge_sigma)
    noise_blurred = normalize_array_to_unity(noise_blurred)

    sorted_gray_scales = np.sort(noise_blurred.flatten())

    clip_index = int(density * np.size(sorted_gray_scales))
    clipval = sorted_gray_scales[clip_index]

    clipped = np.zeros_like(noise_blurred)
    clipped[noise_blurred > clipval] = 1.0

    speckle = gaussian_filter(clipped, sigma=blur_sigma) * -1. + 1.
    return speckle
