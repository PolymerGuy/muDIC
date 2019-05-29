import logging
import numpy as np
import matplotlib.pyplot as plt
import muDIC.vlab as vlab
import muDIC as dic

"""
This example case runs an experiment where a deformation gradient is used
to deform a synthetically generated speckle, the speckle is then down sampled by a factor of four
and sensor artifacts are included.

The analysis is then performed and the resulting deformation gradient field is compared to the
one used to deform the images
"""

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
show_results = False


# Define the image you want to analyse
n_imgs = 2
image_shape = (500, 500)
downsample_factor = 4
super_image_shape = tuple(dim * downsample_factor for dim in image_shape)

# Make a speckle image
speckle_image = vlab.rosta_speckle(super_image_shape, dot_size=4, density=0.5, smoothness=2.0)


# Make an image deformed
F = np.array([[1.01,0],[0.01,1.0]])
image_deformer = vlab.imageDeformer_from_defGrad(F)

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

# Now, make a mesh. Make sure to use enough elements
mesher = dic.Mesher(deg_n=3, deg_e=3)
#mesh = mesher.mesh(image_stack) # Use this if you want to use a GUI
mesh = mesher.mesh(image_stack,Xc1=50,Xc2=450,Yc1=50,Yc2=450,n_ely=8,n_elx=8, GUI=False)


# Prepare the analysis input and initiate the analysis
input = dic.DICInput(mesh, image_stack)
input.tol = 1e-6

dic_job = dic.DICAnalysis(input)
results = dic_job.run()

# Calculate the fields for later use
fields = dic.Fields(results, seed=101)

# We will now compare the results from the analysis to the deformation gradient which the image was deformed by

if show_results:
    plt.figure()
    plt.imshow(F[0,0] - fields.F()[0, 0,0, :, :, 1], cmap=plt.cm.magma)
    plt.xlabel("Element e-coordinate")
    plt.ylabel("Element n-coordinate")
    plt.colorbar()
    plt.title("Difference in deformation gradient component 0,0 within the element")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #line1 = ax1.plot(res_field[:, 50], label="correct")
    line2 = ax1.plot(fields.F()[0, 0,0, :, 50, 1], label="DIC")
    ax1.set_xlabel("element e-coordinate")
    ax1.set_ylabel("Deformation gradient component 0,0 []")

    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    line3 = ax2.plot(F[0,0] - fields.F()[0, 0,0, :, 50, 1], "r--", label="difference")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Deviation []")
    plt.title("Deformation gradient component 0,0")

    fig1.legend()
    plt.show()
