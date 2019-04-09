from functools import partial

import numpy as np


def gaussian_noise_model(size, sigma):
    return np.random.normal(0.0, sigma, size)


def noise_injector_additive(image, noise_generator):
    mean = np.average(image)

    image_noisy = image + mean * noise_generator(image.shape)
    return image_noisy


def noise_injector(noise_model="gaussian", sigma=0.0):
    """Noise injector

    The noise injector returns a function which when passed an image, adds noise to the image.

    The supported noise models are:
        * Gaussian additive noise

    Example
    -------
    Let us make an noise injector which adds gaussian noise with a standard deviation of 1.

    >>> from muDIC import vlab
    >>> import numpy as np
    >>> img = np.ones((100,100))
    >>> noise_injector = vlab.noise_injector("gaussian", sigma=.1)
    >>> noisy_image = noise_injector(img)


    Note
    ----
    More noise models should be added later

    Parameters
    ----------
    noise_model : string
        The noise model to be used. Eg. "gaussian"
    sigma : float
        The standard deviation of the noise distribution
    """
    if noise_model == "gaussian":
        noise_generator = partial(gaussian_noise_model, sigma=sigma)

    return partial(noise_injector_additive, noise_generator=noise_generator)
