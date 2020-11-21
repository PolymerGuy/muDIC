# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import logging
import numpy as np
import matplotlib.pyplot as plt
import muDIC.vlab as vlab
import muDIC as dic

"""
This example case runs an experiment where only rigid body motion is present.
This demostrated the "multi-scale" functionality, which allows for an analysis to be performed with a single
element to produce initial conditions for a more refined mesh.

A rigid body displacement of 10 pixels is used to deform the image and the analysis will not run
without getting initial conditions from a coarser mesh.

The analysis is then performed and the resulting displacement field is compared to the
one used to deform the images
"""

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
show_results = False

# Define the image you want to analyse
n_imgs = 2
image_shape = (500, 500)

# Make a speckle image
speckle_image = vlab.rosta_speckle(image_shape, dot_size=4, density=0.5, smoothness=2.0)

displacement_function = vlab.deformation_fields.rigid_body_x

# Make an image deformed
image_deformer = vlab.imageDeformer_from_uFunc(displacement_function, shift=10.)

# Make an synthetic image generation pipeline
image_generator = vlab.SyntheticImageGenerator(speckle_image=speckle_image, image_deformer=image_deformer, n=n_imgs)
# Put it into an image stack
image_stack = dic.ImageStack(image_generator)

# Now, make a mesh. Make sure to use enough elements
mesher = dic.Mesher(type="q4")
#mesh = mesher.mesh(image_stack)    # Use this if you want to mesh with a GUI
mesh = mesher.mesh(image_stack,Xc1=50,Xc2=450,Yc1=50,Yc2=450,n_ely=10,n_elx=10, GUI=False)


# Perform a pre-run with a single spline element
single_elm_mesh = mesh.single_element_mesh(deg_n=1,deg_e=1)

input_prerun = dic.DICInput(single_elm_mesh, image_stack)
input_prerun.tol = 1e-6

dic_job = dic.DICAnalysis(input_prerun)
results = dic_job.run()

x_nodes,y_nodes = dic.mesh.initial_conds_from_analysis(single_elm_mesh, mesh, results)

# Perform the full analysis using the results from the previous analysis as initial values
input = dic.DICInput(mesh, image_stack)
input.tol = 1e-6
input.node_hist = np.array([x_nodes,y_nodes])

dic_job = dic.DICAnalysis(input)
results = dic_job.run()

# Calculate the fields for later use
fields = dic.Fields(results, seed=101)

# We will now compare the results from the analysis to the displacement which the image was deformed by
# We do this by evaluating the same function as used to deform the images.
# First we find the image coordinates of the image
xs, ys = dic.utils.image_coordinates(image_stack[0])

# We then find the displacement components for each image coordinate
u_x, u_y = displacement_function(xs, ys, shift=10.)

# We now need to find the material points used in the DIC analysis, and extract the corresponding
# correct displacement values
e = fields.coords()[0, 1, :, :, 1]
n = fields.coords()[0, 0, :, :, 1]
res_field = dic.utils.extract_points_from_image(u_x, np.array([e, n]))


if show_results:

    # Now, lets just plot the results!
    plt.figure()
    plt.imshow(u_x, cmap=plt.cm.magma)
    plt.xlabel("Image X-coordinate")
    plt.ylabel("Image Y-coordinate")
    plt.colorbar()
    plt.title("The imposed displacement field component in the X-direction")

    plt.figure()
    plt.imshow(fields.disp()[0, 0, :, :, 1]-res_field, cmap=plt.cm.magma)
    plt.xlabel("Element e-coordinate")
    plt.ylabel("Element n-coordinate")
    plt.colorbar()
    plt.title("Difference in displacement value within the element")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    line1 = ax1.plot(res_field[:, 5], label="correct")
    line2 = ax1.plot(fields.disp()[0, 0, :, 5, 1], label="DIC")
    ax1.set_xlabel("element e-coordinate")
    ax1.set_ylabel("Displacement [pixels]")

    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    line3 = ax2.plot(res_field[:, 5] - fields.disp()[0, 0, :, 5, 1], "r--", label="difference")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Deviation [pixels]")
    plt.title("Displacement field values along the center of the field")

    fig1.legend()
    plt.show()
