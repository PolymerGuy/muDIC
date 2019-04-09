Mesher
=======================================
The mesher package contains everything you need for generation
of simplistic structured meshes. Let us first import the tools::

    import muDIC as dic



Generating a mesh from an image stack
-------------------------------------
In order to generate a mesh, you first need to instanciate a mesher object::

    mesher = dic.Mesher()


Mesher can take a set of settings such as polynomial order and pre-defined knot vectors.
If none are given, it uses the default fisrt order polynomials.

Now, let us make a mesh on the image_stack object we have made earlier::

    mesh = mesher.mesh(image_stack)

A GUI will now pop up, looking something like this:

.. image:: ../examples/figures/mesher.png
   :scale: 100 %
   :alt: The mesher GUI
   :align: center

You can now drag a rectangle over the region you want to cover by the mesh.
To manipulate the mesh, you can use:

*   A,D: add or remove element in the horizontal direction
*   W,X: add or remove element in the vertical direction
*   arrow keys: move the mesh one pixel in the direction of the arrow



If everything is working properly, a small matplotlib-GUI will pop up.
If you dont want to use a GUI, you can set GUI=False and give the numeber of control points 
and the coordinates of the corners manually.

The mesh object is now ready for use.


Manipulating a mesh
-------------------

The mesh-object has a set of methods for manipulating the mesh after it has been created.
We can use them to translate the mesh five pixels in the X-direction::


    mesh.move((5,0))


or scale the mesh by a factor of 1.2::

    mesh.scale(1.2)

