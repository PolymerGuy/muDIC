############
Virtual lab
############

.. toctree::
   :maxdepth: 3
   :hidden:

   speckle_gen
   downsampling
   noise
   deformation
   virtualTest

Virtual lab is a small collection of tools which can be combined to perform virtual tensile tests on speckles,
including sensor artifacts and noise.

The test suite includes:
 * Speckle generation tools
    * Perlin noise based speckles
    * Drop spray based speckles
    * Additional tools for blurring etc.

 * Down sampling tools:
    * Down sampling based on bi-cubic spline interpolation
    * Supports different down sampling factors
    * Supports different fill-factors
    * Supports random pixel offsets

 * Noise injection tools:
    * Gaussion noise injection

 * Image deformation tools:
    * Arbitrary deformation gradients
    * Deformation gradient fields

 * Virtual test class
    * Supports all features out of the box









