import logging
from functools import partial

import numpy as np

from ..utils import find_borders, find_inconsistent

np.seterr(invalid='raise')


class Reference(object):
    def __init__(self, Nref, I0, K, B, n_pixels, e, n, image_id=None):
        """
        Reference container

        Contains all neccessary data related to the reference image.

        Parameters
        ----------
        Nref : ndarray
            The shape function of the element evaluated at all image coordinates
        I0 : ndarray
            The pixel values for all pixels covered by the mesh
        K : ndarray.
            Matrix used during the correlation process
        B : ndarray
            Matrix used during the correlation process
        n_pixels : int
            The number of pixels covered by the mesh
        e : ndarray.
            The e coodinate to the pixels covered by the mesh
        n : ndarray
            The n coodinate to the pixels covered by the mesh
        image_id : int, optional
            The id for the image used to make the reference

        Returns
        -------
        Refernce object


        Notes
        -----
        This constructor does not do any type of typechecking
        """
        self.Nref_stack = Nref
        self.I0_stack = I0
        self.K = K
        self.B_stack = B
        self.n_pixels = n_pixels
        self.e = e
        self.n = n
        self.image_id = image_id


def find_active_pixels(e_coord, n_coord, e_coord_inc, n_coord_inc, min_inc=1e-6):
    """
    Finds active pixels

    Returns a boolean array identifying which pixels have increments larger than the tolerance
    and within the bounds [0,1] of the frame.


    Parameters
    ----------
    e_coord : ndarray
        The e coordinates of the pixel
    n_coord : ndarray
        The n coordinates of the pixel
    e_coord_inc : ndarray.
        The e coordinate increment of the pixel
    n_coord_inc : ndarray
        The n coordinate increment of the pixel
    min_inc : float
        The increment size which defines convergence

    Returns
    -------
    active_pixels : ndarry
        The active pixels given as a 1d boolean array


    Notes
    -----
    """
    e_coord_inbound = np.logical_and(e_coord > 0., e_coord < 1.)
    n_coord_inbound = np.logical_and(n_coord > 0., n_coord < 1.)
    not_converged = np.logical_or(np.abs(e_coord_inc) > min_inc, np.abs(n_coord_inc) > min_inc)

    return np.logical_and(np.logical_and(e_coord_inbound, n_coord_inbound), not_converged)


def clip_args(func, arg1, arg2, bounds=(0., 1.)):
    """
    Clip the arguments to bounds

    Return the results of the function where the clipped arguments have been used.
    Arguments below the lower bound are set to the lower bound and the arguments
    above the upper bound are set to the upper bound.

    Parameters
    ----------
    func : func(arg1,arg2)
        The function which the clipped arguments are passed to
    arg1 : ndarray
        1D array with floats.
    arg2 : ndarray.
        1D array with floats.
    bounds : tuple, optional
        The bounds that the arguments are limited to.

    Returns
    -------
    clipped_func : func(arg1_clipped,arg2_clipped)
        The results of the function where the clipped agruments have been applied.


    Notes
    -----
    This function does not do any type of typechecking
    """
    upper_bound = bounds[1]
    lower_bound = bounds[0]

    arg1_inbound = arg1.copy()
    arg2_inbound = arg2.copy()

    arg1_inbound[arg1 < lower_bound] = lower_bound
    arg1_inbound[arg1 > upper_bound] = upper_bound

    arg2_inbound[arg2 < lower_bound] = lower_bound
    arg2_inbound[arg2 > upper_bound] = upper_bound

    return func(arg1_inbound, arg2_inbound)


def identify_pixels_within_frame(xnod, ynod, elm, over_sampling=1.1):
    """
        Identify pixels covered by an element frame.

        Returns the coordinates of the covered pixels in the image frame and an estimate of the coordinates
        in the element frame.
        This is done by evaluating the element shape functions on a denser grid than the
        image grid, rounds the indices to nearest integer and removes duplicates.
        The element cordinates to the corresponding pixels is then obtained from the same mask.

        Parameters
        ----------
        xnod : ndarray
            1D array with floats.
            The x coordinates of the control points.
        ynod : ndarray
            1D array with floats.
            The y coordinates of the control points.
        elm : interpolator object.
            The interpolator object provides the shape functions used to calculate the coordinates within the element.
        over_sampling : float, optional
            The degree of oversampling used to find the pixels.

        Returns
        -------
        pixel_x : ndarray
            The x-coordinates of the pixels covered by the element
        pixel_y : ndarray
            The y-coordinates of the pixels covered by the element
        pixel_es : ndarray
            The elemental e-coordinates of the pixels covered by the element
        pixel_ns : ndarray
            The elemental n-coordinates of the pixels covered by the element


        Notes
        -----
        There is no guarantee that all pixels are found, so when in doubt, increase the over_sampling factor.

        """
    x_min, x_max = find_borders(xnod)
    y_min, y_max = find_borders(ynod)

    # Calculate coordinates (e,n) covered by the element on a fine grid
    n_search_pixels = np.int(over_sampling * max((x_max - x_min), y_max - y_min))
    es, ns = np.meshgrid(np.linspace(0., 1., n_search_pixels), np.linspace(0., 1., n_search_pixels))
    es = es.flatten()
    ns = ns.flatten()

    pixel_xs = np.dot(elm.Nn(es, ns), xnod)
    pixel_ys = np.dot(elm.Nn(es, ns), ynod)

    pixel_xs_closest = np.around(pixel_xs).astype(np.int)
    pixel_ys_closest = np.around(pixel_ys).astype(np.int)

    xs_ys = np.stack([pixel_xs_closest, pixel_ys_closest], axis=0)
    xs_ys_unique, unique_inds = np.unique(xs_ys, return_index=True, axis=1)

    pixel_x = xs_ys_unique[0, :].astype(np.float64)
    pixel_y = xs_ys_unique[1, :].astype(np.float64)

    pixel_es = es[unique_inds].astype(np.float64)
    pixel_ns = ns[unique_inds].astype(np.float64)

    return pixel_x, pixel_y, pixel_es, pixel_ns


def find_covered_pixel_blocks(node_x, node_y, elm, max_iter=200, block_size=1e7, tol=1.e-6):
    """
    Find element coordinates to all pixels covered by the element.

    Returns the coordinates of the covered pixels in the image coordinates and in the element coordinates.
    This is done by first identifiying the pixels within the frame and then finding the corresponding
    element coordinates by using a modified Newton-Raphson scheme. For reduced memory usage, the
    image covered by the element is subdivided into blocks.

    Parameters
    ----------
    node_x : ndarray
        1D array with floats.
        The x coordinates of the control points.
    node_y : ndarray
        1D array with floats.
        The y coordinates of the control points.
    elm : interpolator object.
        The interpolator object provides the shape functions used to calculate the coordinates within the element.
    max_iter : int, optional
        The maximum allowed number of iterations
    block_size :int, optional
        The maximum number of elements in each block
        The number of elements are N-pixels X N-Control points

    tol : float, optional
        The convergence criteria

    Returns
    -------
    pixel_x : ndarray
        The x-coordinates of the pixels covered by the element
    pixel_y : ndarray
        The y-coordinates of the pixels covered by the element
    pixel_es : ndarray
        The elemental e-coordinates of the pixels covered by the element
    pixel_ns : ndarray
        The elemental n-coordinates of the pixels covered by the element


    Notes
    -----

    """

    logger = logging.getLogger(__name__)

    # e and n are element coordinates
    found_e = []
    founc_n = []
    # x and y are the corresponding image coordinates
    found_x = []
    found_y = []

    # These are just estimates
    pix_Xs, pix_Ys, pix_es, pix_ns = identify_pixels_within_frame(node_x, node_y, elm)

    # Split into blocks
    n_pix_in_block = block_size / np.float(len(node_x))
    num_blocks = np.ceil(len(pix_es) / n_pix_in_block).astype(np.int)
    logger.info("Splitting in %s blocks:" % str(num_blocks))

    pix_e_blocks = np.array_split(pix_es, num_blocks)
    pix_n_blocks = np.array_split(pix_ns, num_blocks)

    pix_X_blocks = np.array_split(pix_Xs, num_blocks)
    pix_Y_blocks = np.array_split(pix_Ys, num_blocks)

    for block_id in range(num_blocks):
        e_coord = pix_e_blocks[block_id]
        n_coord = pix_n_blocks[block_id]

        X_coord = pix_X_blocks[block_id]
        Y_coord = pix_Y_blocks[block_id]

        # Empty increment vectors
        n_coord_inc = np.zeros_like(n_coord)
        e_coord_inc = np.zeros_like(e_coord)

        # Pre-calculate the gradients. This results in a modified Newton scheme
        dxNn = clip_args(elm.dxNn, e_coord, n_coord)
        dyNn = clip_args(elm.dyNn, e_coord, n_coord)

        for i in range(max_iter):

            Nn = clip_args(elm.Nn, e_coord, n_coord)

            n_coord_inc[:] = (Y_coord - np.dot(Nn, node_y) - np.dot(dxNn, node_y) * (
                    X_coord - np.dot(Nn, node_x)) / (np.dot(dxNn, node_x))) / (
                                     np.dot(dyNn, node_y) - np.dot(dxNn, node_y) * np.dot(
                                 dyNn, node_x) / np.dot(dxNn, node_x))

            e_coord_inc[:] = (X_coord - np.dot(Nn, node_x) - np.dot(dxNn, node_x) *
                              n_coord_inc) / np.dot(dxNn, node_x)

            e_coord[:] += e_coord_inc
            n_coord[:] += n_coord_inc

            active_pixels = find_active_pixels(e_coord, n_coord, e_coord_inc, n_coord_inc, tol)

            if not np.any(active_pixels):
                logger.info('Pixel coordinates found in %i iterations', i)

                epE_block, nyE_block, Xe_block, Ye_block = map(
                    partial(np.delete, obj=find_inconsistent(e_coord, n_coord)),
                    [e_coord, n_coord, X_coord, Y_coord])

                found_e.append(epE_block)
                founc_n.append(nyE_block)
                found_x.append(Xe_block.astype(np.int))
                found_y.append(Ye_block.astype(np.int))

                break
            if (i + 1) == max_iter:
                raise RuntimeError("Did not converge in %i iterations" % max_iter)

    return found_e, founc_n, found_x, found_y


def generate_reference(nodal_position, mesh, image, settings, image_id=None):
    """
    Generates a Reference object

    The Reference object contains all internals that will be used during the correlation procedure.


   Parameters
   ----------
   nodal_position : ndarray
       2D array with floats.
       The coordinates of the control points.
   mesh : Mesh object
       Mesh definitions
   image : ndarray
       image as an 2d array
   image_id : int, optional
       The image id, stored for further reference


   Returns
   -------
   reference : Reference
       The Reference object

   Notes
   -----
   The current implementation is slow but not very memory intensive.

   Theory
   -----


   """
    logger = logging.getLogger()
    elm = mesh.element_def

    node_xs = nodal_position[0]
    node_ys = nodal_position[1]

    img_grad = np.gradient(image)

    try:
        pixel_e_blocks, pixel_n_blocks, pixel_x_blocks, pixel_y_blocks = find_covered_pixel_blocks(node_xs,
                                                                                                   node_ys,
                                                                                                   elm,
                                                                                                   block_size=settings.block_size)

        num_blocks = len(pixel_e_blocks)
        num_pixels = np.sum([block.size for block in pixel_e_blocks])

        K = np.zeros((2 * mesh.n_nodes, num_pixels), dtype=settings.precision)
        A = np.zeros((mesh.n_nodes * 2, mesh.n_nodes * 2), dtype=settings.precision)

        img_covered = image[np.concatenate(pixel_y_blocks), np.concatenate(pixel_x_blocks)]

        # Calculate A = B^T * B
        for block_id in range(num_blocks):
            block_len = pixel_e_blocks[block_id].shape[0]
            B = np.zeros((block_len, 2 * mesh.n_nodes), dtype=settings.precision)

            # Weight the image gradients with the value of the shape functions
            B[:, :elm.n_nodes] = (
                    img_grad[1][pixel_y_blocks[block_id], pixel_x_blocks[block_id]][:, np.newaxis] * elm.Nn(
                pixel_e_blocks[block_id], pixel_n_blocks[block_id]))

            B[:, elm.n_nodes:] = (
                    img_grad[0][pixel_y_blocks[block_id], pixel_x_blocks[block_id]][:, np.newaxis] * elm.Nn(
                pixel_e_blocks[block_id], pixel_n_blocks[block_id]))
            A += np.dot(B.transpose(), B)

        pixel_ind = 0
        pixel_ind_last = 0

        # Determine K
        for block_id in range(num_blocks):
            block_len = pixel_e_blocks[block_id].shape[0]

            B = np.zeros((2 * mesh.n_nodes, block_len), dtype=settings.precision)

            pixel_ind += block_len

            # Weight the image gradients with the value of the shape functions
            # TODO: This operation is duplicate
            B[:elm.n_nodes, :] = (
                    img_grad[1][pixel_y_blocks[block_id], pixel_x_blocks[block_id]][:, np.newaxis] * elm.Nn(
                pixel_e_blocks[block_id], pixel_n_blocks[block_id])).transpose()

            B[elm.n_nodes:, :] = (
                    img_grad[0][pixel_y_blocks[block_id], pixel_x_blocks[block_id]][:, np.newaxis] * elm.Nn(
                pixel_e_blocks[block_id], pixel_n_blocks[block_id])).transpose()

            K_block = np.linalg.solve(A, B)

            K[:, pixel_ind_last:pixel_ind] = K_block

            pixel_ind_last = pixel_ind

        # Remove for reduced memory usage
        del B, K_block

        Nn = elm.Nn(np.concatenate(pixel_e_blocks), np.concatenate(pixel_n_blocks)).transpose()
        pixel_es = np.concatenate(pixel_e_blocks)
        pixel_ns = np.concatenate(pixel_n_blocks)


    except Exception as e:
        logger.exception(e)
        raise RuntimeError('Failed to generate reference')

    return Reference(Nn, img_covered, K, None, num_pixels, pixel_es, pixel_ns,
                     image_id=image_id)
