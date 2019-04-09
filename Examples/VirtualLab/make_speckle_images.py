import logging
import numpy as np
import matplotlib.pyplot as plt
import muDIC.vlab as vlab
import muDIC as dic

"""
This example generates speckle images where a bi-harmonic deformation field is used
to deform a synthetically generated speckle, the speckle is then downsampled by a factor of four
and sensor artifacts are included.

"""

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Define the image you want to analyse
n_imgs = 2
image_shape = (500, 500)
downsample_factor = 4
super_image_shape = tuple(dim * downsample_factor for dim in image_shape)

# Make a speckle image
speckle_image = vlab.rosta_speckle(super_image_shape, dot_size=4, density=0.5, smoothness=2.0)

displacement_function = vlab.deformation_fields.harmonic_bilat

# Make an image deformed
image_deformer = vlab.imageDeformer_from_uFunc(displacement_function, omega=3 * np.pi, amp=2.0)

# Make an image down-sampler including downscaling, fill-factor and sensor grid irregularities
downsampler = vlab.Downsampler(image_shape=super_image_shape, factor=downsample_factor, fill=.95,
                               pixel_offset_stddev=0.05)

# Make a noise injector producing 2% gaussian additive noise
noise_injector = vlab.noise_injector("gaussian", sigma=.02)

# Make an synthetic image generation pipeline
image_generator = vlab.SyntheticImageGenerator(speckle_image=speckle_image, image_deformer=image_deformer,
                                               downsampler=downsampler, noise_injector=noise_injector, n=n_imgs)
# Put it into an image stack
image_stack = dic.ImageStack(image_generator)

for index,speckle in enumerate(image_stack):
    plt.imsave("speckle%i.tif"%index,speckle,cmap=plt.cm.gray)