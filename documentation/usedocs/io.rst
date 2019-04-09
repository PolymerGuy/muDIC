IO tools
=======================================

This package contains the tools you need for importing images.
Lets import all tools first::

    import muDIC as dic

Importing images from a folder
---------------------------------

Say you have a folder with a set of .png images which you want to use for your DIC analysis.
We can then import them all into an image stack::

    path = r"/the/path/to/the/images/"
    image_stack = dic.image_stack_from_folder(path,filetype=".png")

Creating an image stack from a list of images
---------------------------------------------

Say you have imported a list of images from somewhere which you want to use for your DIC analysis.
We can then import them all into an image stack::

    path = r"/the/path/to/the/images/"
    image_stack = dic.image_stack_from_list(list_of_images)


Manipulating the image stack
----------------------------
The image_stack object has a set of methods for manipulating the behaviour of the stack.
Let us skip the first 10 images::

    image_stack.skip_frames(range(10))

and for some strange reason reverse the order of the images::

    image_stack.reverse()

Adding a filter to the image stack
---------------------------------
In many applications, filtering of the images prior to the DIC analysis ca be attractive.

Let us now a add gaussian blur filter with a standard deviation of one pixel to all images::

    image_stack.add_filter(dic.filters.gaussian_lowpass,sigma=1.0)


