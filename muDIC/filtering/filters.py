import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import median_filter


def highpass_gaussian(image, sigma=2.0):
    return image - nd.gaussian_filter(image, sigma=sigma)


def lowpass_gaussian(image, sigma=2.0):
    return nd.gaussian_filter(image, sigma=sigma)


def homomorphic_median(image, sigma=10):
    log_img = np.log(image)
    return np.exp(log_img - median_filter(log_img, int(sigma)))
