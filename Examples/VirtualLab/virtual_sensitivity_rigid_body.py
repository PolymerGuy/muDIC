import muDIC as dic
from muDIC import vlab
import matplotlib.pyplot as plt
import logging
import numpy as np

"""
In this example we investigate how small displacements we can detect for a given noise level.
We do this by making an speckle image, shifting it with a small displacement and run the 
correlation routines to see whether we can measure the displacement. We do this for different noise levels.

As noise is a stochastic entity, we do a number of realisations for each noise amplitude.

We use a fourier shift to shift the image by a given value, avoiding interpolation. <-Good!
"""

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# We use only two images, a reference and a shifted image
n_imgs = 2

# We use the rosta algorithm to make a speckle pattern
# The image size is set to a convenient size of 500 x 500 pixels
# The greyscales of the synthetic image spans [0,1]
img = vlab.rosta_speckle((500, 500), dot_size=4, density=0.5, smoothness=3.0)

# We can visualize the speckle
plt.imshow(img,cmap=plt.cm.gray)
plt.show(block=True)

# Here are the settings for the analysis
# We run the analysis for this range of gaussian noise standard deviations
noise_stds = np.linspace(0.0, 0.03, 5)
# We do a number of realisations for each noise standard deviation
repts = 100
# We use a mesh of NxN elements
num_elms = 20
# Shift amplitude along the X-axis
shift_amp = 0.1

# Instantiate some list where we can store the results
std = []
mean = []

for noise_std in noise_stds:
    # Instantiate some lists where we store the results for each realisation of the noise spectra
    std_rep = []
    mean_rep = []

    for rep in range(repts):
        # Instantiate a noise injector with the noise standard deviation
        noise_injector = vlab.noise.noise_injector("gaussian", sigma=noise_std)

        # Instantiate an image deformer
        image_deformer = vlab.image_deformer.imageDeformer_rigid_body(amp=(shift_amp, 0.))

        # Shift the images. Note that the first one is not shifted
        shifted_images = image_deformer(img, n_imgs)

        # We now add noise to the images
        shifted_noisy_images = [noise_injector(image) for image in shifted_images]

        # And put them into a stack which is the formatting needed for the DIC analysis
        image_stack = dic.image_stack_from_list(shifted_noisy_images)

        # Generate mesh
        # We here use Q4 elements
        mesher = dic.Mesher(type="q4")
        # If you want to use a GUI, set GUI=True
        mesh = mesher.mesh(image_stack, Xc1=20, Xc2=480, Yc1=20, Yc2=480, n_ely=num_elms, n_elx=num_elms, GUI=False)

        # Instantiate settings object and set some settings manually
        settings = dic.DICInput(mesh, image_stack)
        # We allow the solver to use as many iterations as it needs
        settings.maxit = 100
        # This tolerance is the larges increment in nodal position [pixels]
        settings.tol = 1e-6
        # We use fourth order splines for grey-scale interpolation, minimizing interpolation bias
        settings.interpolation_order = 4

        # Instantiate job object
        job = dic.DICAnalysis(settings)
        # Running DIC analysis
        dic_results = job.run()

        # Calculate field values
        fields = dic.post.viz.Fields(dic_results, upscale=1)
        # Calculate the mean value of the displacement field
        mean_rep.append(np.mean(fields.disp()[0, 0, :, :, -1]))
        # Calculate the standard deviation of the displacement field
        std_rep.append(np.std(fields.disp()[0, 0, :, :, -1]))

    # Store the results for this realisation of the noise spectra
    mean_rep = np.array(mean_rep)
    std_rep = np.array(std_rep)
    mean.append(mean_rep)
    std.append(std_rep)

# Determine the mean values of the realisations and the standard deviation
mean = np.array(mean)
mean_mean = np.mean(mean, axis=1)
mean_std = np.std(mean, axis=1)

std = np.array(std)
std_mean = np.mean(std, axis=1)
std_std = np.std(std, axis=1)

noise_stds = np.array(noise_stds)

plt.figure()
plt.plot(noise_stds, mean_mean, '-', color="blue")
plt.fill_between(noise_stds, mean_mean - mean_std, mean_mean + mean_std, color="blue", alpha=0.3, linewidth=0.)

plt.xlim(left=0.0, right=np.max(noise_stds))
plt.xlabel("Grey scale noise standard deviation [%]")
plt.ylabel("Mean of displacement amplitude [-]", color="blue")

ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Standard deviation of displacement field [-]', color="red")  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor="red")

ax2.plot(noise_stds, std_mean, '-', color="red")
ax2.fill_between(noise_stds, std_mean - std_std, std_mean + std_std, color="red", alpha=0.3, linewidth=0.)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
