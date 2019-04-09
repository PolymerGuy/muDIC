from __future__ import absolute_import

import logging
from copy import copy

import numpy as np
import scipy.ndimage as nd

from .reference import generate_reference as generate_reference
from ..IO.image_stack import ImageStack
from ..elements.fieldInterpolator import FieldInterpolator
from ..mesh.meshUtilities import Mesh
from ..solver.custom_exceptions import DidNotConverge
from ..utils import convert_to_img_frame, find_element_borders


def correlate_img_to_ref(node_pos, mesh, img, ref, settings):
    """
   Correlate an image to a reference
    The routine identifies the part of the image covered by the mesh and
    tries to perform image correlation on this part of the image.

    Parameters
    ----------
    node_pos : ndarray
       The position of the nodes
    mesh : Mesh
       The mesh object
    img : ndarray
       2d array containing the image
    ref : Reference
       The reference object
    settings : DICInput
       The settings which will be used during the analysis
   Returns
   -------
   updated node positions, current pixel values

   NOTES
   -------
   The function extracts a rectangular region of the image covered by the element, which may be very large
   if the mesh is tilted. This would reduce the performance of the routine
    """
    element_borders = find_element_borders(node_pos, mesh)

    image_frame, node_pos_img_coords = convert_to_img_frame(img, node_pos, mesh, element_borders, settings)

    node_position_increment, Ic = correlate_frames(node_pos_img_coords, mesh, image_frame, ref, settings)

    node_position_new = node_pos + node_position_increment
    return node_position_new, Ic


def correlate_frames(node_pos, mesh, img, ref, settings):
    """
    Parameters
    ----------
    node_pos : ndarray
       The position of the nodes
    mesh : Mesh
       The mesh object
    img : ndarray
       2d array containing the image frame
    ref : Reference
       The reference object
    settings : DICInput
       The settings which will be used during the analysis
   Returns
   -------
   updated node positions, current pixel values
    """

    logger = logging.getLogger(__name__)

    node_pos = np.copy(node_pos).astype(settings.precision)

    # Declare empty arrays
    pixel_pos = np.zeros((2, ref.n_pixels), dtype=settings.precision)
    dnod_x = np.zeros(mesh.n_nodes * 2)

    image_filtered = nd.spline_filter(img, order=3).transpose()

    for it in range(settings.maxit):

        # Find nodal positions within ROI
        np.dot(node_pos, ref.Nref_stack, out=pixel_pos)

        # Find pixel values for current coordinates
        Ic = nd.map_coordinates(image_filtered, pixel_pos, order=3, prefilter=False)

        # Calculate position increment as (B^T B)^-1 * (B^T*dIk) "Least squares solution"
        dnod = np.dot(ref.K, ref.I0_stack - Ic)

        # Add increment to nodal positions
        node_pos[0, :] += dnod[:mesh.n_nodes]
        node_pos[1, :] += dnod[mesh.n_nodes:]

        dnod_x += dnod

        # Check for convergence
        if np.max(np.abs(dnod)) < settings.tol:
            logger.info('Frame converged in %s iterations', it)
            return np.array((dnod_x[:mesh.n_nodes], dnod_x[mesh.n_nodes:])), Ic

        # Reset array values
    raise DidNotConverge('No convergence')


def store_stripped_copy(reference_generator, storage):
    """
   Store a stripped copy of the reference object
    The stored reference object is stripped from all resource intensive variables

   Parameters
   ----------
   reference_generator : func
       The reference generator function
   storage : list
       The list where the stripped references are appended
   Returns
   -------
   reference_generator
    """

    def wrapper(*args, **kwargs):
        ref = reference_generator(*args, **kwargs)
        ref_light = copy(ref)

        # Remove the heavy fields
        ref_light.Nref_stack = None
        ref_light.B_stack = None
        ref_light.K = None
        storage.append(ref_light)
        return ref

    return wrapper


def correlate(inputs):
    """
   Main correlation routine
    This routine manages result storage, reference generation and
     the necessary logic for handling convergence issues.

   Parameters
   ----------
   inputs : DIC_input object
       The input object containing all necessary data for performing a DIC analysis.

   Returns
   -------
   node_coords, reference_stack, Ic_stacks
    """

    logger = logging.getLogger(__name__)

    mesh = inputs.mesh
    images = inputs.images
    settings = inputs

    # Do the initial setup

    images.image_reader.precision = settings.precision

    Ic_stacks = list()
    reference_stack = list()
    node_position_t = list()

    if settings.store_internals:
        gen_ref = store_stripped_copy(generate_reference, storage=reference_stack)
    else:
        gen_ref = generate_reference

    if settings.node_hist:
        node_coords = np.array(settings.node_hist, dtype=settings.precision)[:, :, 0]
    else:
        node_coords = np.array((mesh.xnodes, mesh.ynodes), dtype=settings.precision)

    reference = gen_ref(node_coords, mesh, images[0], settings, image_id=0)

    # Correlate the image frames

    try:
        for image_id in range(settings.max_nr_im):
            logger.info('Processing frame nr: %i', image_id)

            if settings.node_hist:
                node_coords = np.array(settings.node_hist, dtype=settings.precision)[:, :, image_id]

            if image_id in settings.ref_update:
                logger.info('Updating reference at %i', image_id)
                reference = gen_ref(node_coords, mesh, images[image_id - 1], settings, image_id=(image_id - 1))

            img = images[image_id]

            try:
                node_coords, Ic = correlate_img_to_ref(node_coords, mesh, img, reference, settings)

            except DidNotConverge:
                # If the reference is new, there is nothing more to do. Otherwise, update the reference and try again.
                if reference.image_id == image_id - 1:
                    logger.info('Correlation failed at frame %i', image_id)
                    break
                else:
                    logger.info('Updating reference at %i', image_id)
                    reference = gen_ref(node_coords, mesh, images[image_id - 1], settings, image_id - 1)
                    node_coords, Ic = correlate_img_to_ref(node_coords, mesh, img, reference, settings)

            except Exception as e:
                logger.info('Correlation failed with error at frame %i', image_id)
                logger.exception(e)
                break

            if settings.store_internals:
                Ic_stacks.append(Ic)

            node_position_t.append(node_coords)

    finally:
        return np.array(node_position_t), reference_stack, Ic_stacks


class DICAnalysis(object):

    def __init__(self, inputs):

        """
         DIC analysis

        The analysis object verifies and stores the DIC_input object.
        When instantiated, the .run() method can be called, initiating the DIC analysis.

         Parameters
         ----------
         inputs : DIC_input object
             The input object containing all neccessary data for performing a DIC analysis.

         Returns
         -------
         DIC_analysis object


         Examples
         --------
        The following example runs a virtual experiment

            >>> import muDIC as dic
            >>> import numpy as np
            >>> import muDIC.vlab as vlab

            >>> image_shape = (2000, 2000)
            >>> speckle_image = vlab.rosta_speckle(image_shape, dot_size=4, density=0.32, smoothness=2.0, layers=4)

            >>> F = np.array([[1.1, .0], [0., 1.0]], dtype=np.float64)
            >>> image_deformer = vlab.imageDeformer_from_defGrad(F)
            >>> downsampler = vlab.Downsampler(image_shape=image_shape, factor=4, fill=0.8, pixel_offset_stddev=0.1)
            >>> noise_injector = vlab.noise_injector("gaussian", sigma=.1)
            >>> image_generator = vlab.VirtualExperiment(speckle_image=speckle_image, image_deformer=image_deformer,
            >>>                              downsampler=downsampler, noise_injector=noise_injector, n=n)
            >>> image_stack = dic.ImageStack(image_generator)

            >>> mesher = dic.Mesher(deg_n=1,deg_e=1)
            >>> mesh = mesher.mesh(image_stack)

            >>> input = muDIC.solver.correlate.DIC_input(mesh, image_stack)
            >>> dic_job = dic.DIC_analysis(input)
            >>> results = dic_job.run()

         """
        self.logger = logging.getLogger()

        self.__input__ = self.__verify_dic_input__(inputs)

    def run(self):
        """
         Run analysis

         Parameters
         ----------

         Returns
         -------
         DIC_output object



         """
        node_x, node_y, reference_stack, Ic_stack = self.__solve__()

        return DICOutput(node_x, node_y, self.__input__, ref_stack=reference_stack, Ic_stack=Ic_stack)

    def get_input(self):
        return self.__input__

    def __solve__(self):
        node_position, reference_stack, Ic_Stack = correlate(self.__input__)
        # TODO: Remove the need of transposing the matrices
        return node_position[:, 0, :].transpose(), node_position[:, 1,
                                                   :].transpose(), reference_stack, Ic_Stack

    @staticmethod
    def __verify_dic_input__(inputs):
        """
        Input type checker. Verifies all input and calculates missing values if they can be deduced from the others.

        """
        inputs_checked = inputs

        if not isinstance(inputs_checked, DICInput):
            raise TypeError('Inputs has to be an instance of the DICInput class')

        if not isinstance(inputs_checked.images, (ImageStack)):
            raise TypeError('Image stack is not an instance of Image_reader')

        if not isinstance(inputs_checked.mesh, Mesh):
            raise TypeError('Mesh should be an instance of Mesh')

        if isinstance(inputs_checked.max_nr_im, int) and inputs_checked.max_nr_im <= len(inputs_checked.images):
            pass
        else:
            inputs_checked.max_nr_im = len(inputs_checked.images)

        if not isinstance(inputs_checked.mesh.element_def, FieldInterpolator):
            raise TypeError('Finite element should be an instance of Finite_Element')
        inputs_checked.elm = inputs_checked.mesh.element_def

        if not isinstance(inputs_checked.ref_update, (list, tuple)):
            raise TypeError('Reference update frames should be specified by a list or tuple')

        if type(inputs_checked.maxit) is not int:
            raise TypeError('Maximum number of iterations should be specified by an integer')

        if type(inputs_checked.pad) is not int:
            raise TypeError('Padding width should be specified by an integer')

        return inputs_checked


class DICInput(object):

    def __init__(self, mesh, image_stack, ref_update_frames=[50, 150], maxit=40, max_nr_im=None, pad=10,
                 store_internals=False, node_hist=None, precision="double", interpolation_order=3, block_size=1e7):

        """
         DIC output container

        This class contains all the necessary inputs for a DIC analysis.

         Parameters
         ----------
         mesh : Mesh object
             The mesh corresponding to the image stack
         image_stack : ImageStack
             The image_stack containing all images for the DIC analysis
         ref_update_frames : list, optional
             A list of indices for the images where a reference update should be performed
         maxit : int, optional
             The maximum allowed number of iterations for the DIC solver before a reference update is performed.
         max_nr_im : int, optional
             The maximum number of images to be analyses
         pad : int, optional
             The amount of padding used around the mesh when the image sub-frame is extracted
         store_internals : bool, optional
             If True, all references are stored and available in the DIC_output
         node_hist : ndarray, optional
             An array containing the nodal positions for each image frame. This is used as initial conditions for the solver
         precision : strings, optional
             The number precision to be used, either "single" or "double".
         interpolation_order : int, optional
             The polynomial order used for image interpolation during the DIC analysis.
         Returns
         -------
         DIC_input object


         Examples
         --------



         """

        self.mesh = mesh
        self.images = image_stack
        self.max_nr_im = max_nr_im
        self.ref_update = ref_update_frames
        self.maxit = maxit
        self.pad = pad
        self.elm = None
        self.tol = 1e-6
        self.store_internals = store_internals
        self.node_hist = node_hist
        self.interpolation_order = interpolation_order
        self.block_size = block_size

        if precision == "single":
            self.precision = np.float32
        else:
            self.precision = np.float64


class DICOutput(object):
    def __init__(self, node_x, node_y, settings, ref_stack=None, Ic_stack=None):
        """
        DIC output container

        This object contains the DIC results

         Parameters
         ----------
         node_x : ndarray
             The nodal x positions for each frame
         node_y : ImageStack
             The nodal y positions for each frame
         settings : list, optional
             The DIC_input object used for the analysis
         ref_stack : int, optional
             A list containing all the references used during the analysis
         Ic_stack : int, optional
             A list containing all the pixel intensities used during the analysis
         Returns
         -------
         DIC_output object
         """

        self.xnodesT = node_x
        self.ynodesT = node_y
        self.crosscorrelation = None
        self.reference = ref_stack
        self.Ic_stack = Ic_stack
        self.settings = settings
        self.settings.images = None
