import muDIC as dic
import logging

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
path = r'./example_data/' # Use this formatting on Linux and Mac OS
#path = r'c:\path\to\example_data\\'  # Use this formatting on Windows

# Generate image instance containing all images found in the folder
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
#images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


# Generate mesh
mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")

# If you want to see use a GUI, set GUI=True below
mesh = mesher.mesh(images,Xc1=316,Xc2=523,Yc1=209,Yc2=1055,n_ely=36,n_elx=9, GUI=False)

# Instantiate settings object and set some settings manually
settings = dic.DICInput(mesh, images)
settings.max_nr_im = 500
settings.ref_update = [15]
settings.maxit = 20
settings.tol = 1.e-6
settings.interpolation_order = 4
# If you want to access the residual fields after the analysis, this should be set to True
settings.store_internals = True

# This setting defines the behaviour when convergence is not obtained
settings.noconvergence = "ignore"

# Instantiate job object
job = dic.DICAnalysis(settings)

# Running DIC analysis
dic_results = job.run()

# Calculate field values
fields = dic.post.viz.Fields(dic_results,upscale=10)

# Show a field
viz = dic.Visualizer(fields,images=images)

# Uncomment the line below to see the results
# viz.show(field="displacement", component = (1,1), frame=-1)

