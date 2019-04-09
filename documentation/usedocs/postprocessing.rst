Post processing
=======================================
After doing a DIC analysis, we are ready to calculate the field variables and 
to visualize the results.

Import the toolkit::

    import muDIC as dic

Calculate fields
----------------

Let's assume we have some dic_results available.

First, we calculate the fields such as deformation gradients and strains::

    fields = dic.Fields(dic_results)

The fields object is lazy and will not calculate anything before the field is quired.

Extract a field variable
------------------------

If you want to extract a fields for use somewhere else, you can do this by::

    true_strain = fields.true_strain()

the true_strain varialble is now a ndarray with the following shape::

    true_strain.shape
    (100,2,2,21,21)

in our example, this shape corresponds to the formatting:
(img_frames,i,j,e,n)
where img_frames is the number of processed images, i and j are the components of the true strain tensor,
and e,n are the iso-parametric element coordinates.

Visualize fields
---------------------------
We can visualize fields manually by using matplotlib or you could use the visualizer included in the toolkit.

Now, lets have a look at the results by using the visualizer::
First, we need to instanciate it::

    viz = dic.visualizer(fields,images=image_stack)

If we provide the images argument, the fields will be overlayed on the images.
Then, we can use the .show method to look at a field for a given frame::

    viz.show(field="True strain", component = (1,1), frame = 45)

This will show us the 11 component of the true strain field at frame 45


