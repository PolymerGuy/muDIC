import muDIC as dic
import logging

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
path = r'./example_data/'

# Generate image instance containing all images found in the folder
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


# Generate mesh
mesher = dic.Mesher(deg_e=3, deg_n=3)
mesh = mesher.mesh(images,Xc1=200,Xc2=1050,Yc1=200,Yc2=650,n_ely=8,n_elx=8, GUI=False)


# Instantiate settings object and set some settings manually
settings = dic.solver.correlate.DICInput(mesh, images)
settings.max_nr_im = 40
settings.ref_update = [15]
settings.maxit = 20
# If you want to access the residual fields after the analysis, this should be set to True
settings.store_internals = False

# Instantiate job object
job = dic.solver.DICAnalysis(settings)

# Running DIC analysis
dic_results = job.run()

# Calculate field values
fields = dic.post.viz.Fields(dic_results)

# Show a field
viz = dic.Visualizer(fields,images=images)
viz.show(field="true strain", component = (1,1), frame = 39)

