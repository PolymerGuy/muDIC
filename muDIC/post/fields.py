import logging

import numpy as np
from scipy.ndimage import map_coordinates

from muDIC.elements import Q4, BSplineSurface
import matplotlib.pyplot as plt

def element_values_to_grid(F, coords, settings):
    # Flatten things form multiple elements to a grid of elements
    grid_shape = (settings.mesh.n_ely, settings.mesh.n_elx)
    n_frames = F.shape[-1]
    F2 = np.zeros(
        (1, 2, 2, settings.mesh.n_elx, settings.mesh.n_ely, F.shape[-1]))
    for i in range(2):
        for j in range(2):
            for t in range(n_frames):
                F2[0, i, j, :, :, t] = F[:, i, j, 0, 0, t].reshape(grid_shape).transpose()

    coords2 = np.zeros(
        (1, 2, settings.mesh.n_elx, settings.mesh.n_ely, F.shape[-1]))
    for i in range(2):
        for t in range(n_frames):
            coords2[0, i, :, :, t] = coords[:, i, 0, 0, t].reshape(grid_shape).transpose()

    return F2, coords2

def element_values_to_grid_with_overlaps(F, coords, settings):
    # Flatten things form multiple elements to a grid of elements

    F = F.copy()
    # Account for multiple samples per element
    n,m = F.shape[-3:-1]
    n_frames = F.shape[-1]

    mesh_shape = (settings.mesh.n_ely, settings.mesh.n_elx)
    F = F.reshape((*mesh_shape,*F.shape[1:]))
    coords = coords.reshape((*mesh_shape,*coords.shape[1:]))

    F2 = np.zeros(
        (1, 2, 2, settings.mesh.n_ely * (n-1)+1, settings.mesh.n_elx * (m-1)+1, F.shape[-1]))
    for i in range(2):
        for j in range(2):
            for t in range(n_frames):
                for k in range(mesh_shape[0]):
                    for l in range(mesh_shape[1]):
                        F2[0, i, j, k*2:k*2+3, l*2:l*2+3, t] = F[k,l, i, j, :, :, t].transpose()

    coords2 = np.zeros(
        (1, 2, settings.mesh.n_ely * (n-1)+1, settings.mesh.n_elx * (m-1)+1, F.shape[-1]))
    for i in range(2):
        for t in range(n_frames):
            for k in range(mesh_shape[0]):
                for l in range(mesh_shape[1]):
                    coords2[0, i, k*2:k*2+3, l*2:l*2+3, t] = coords[k,l, i, :, :, t].transpose()

    return F2, coords2


def upsample_mesh(F, coords, upscale, interpolation_order):
    # TODO: This is broken as upscale=1 looses one row and column!
    n_frames = F.shape[-1]
    elms_y_fine, elms_x_fine = np.meshgrid(np.arange(0, coords.shape[-3] - 1, 1. / upscale),
                                           np.arange(0, coords.shape[-2] - 1, 1. / upscale))

    F3 = np.zeros(
        (1, 2, 2, elms_x_fine.shape[1], elms_x_fine.shape[0], F.shape[-1]))

    coords3 = np.zeros(
        (1, 2, elms_x_fine.shape[1], elms_x_fine.shape[0], F.shape[-1]))

    for i in range(2):
        for t in range(n_frames):
            coords3[0, i, :, :, t] = map_coordinates(coords[0, i, :, :, t],
                                                     [elms_y_fine.flatten(),
                                                      elms_x_fine.flatten()],
                                                     order=interpolation_order).reshape(
                elms_x_fine.shape).transpose()

    for i in range(2):
        for j in range(2):
            for t in range(n_frames):
                F3[0, i, j, :, :, t] = map_coordinates(F[0, i, j, :, :, t],
                                                       [elms_y_fine.flatten(),
                                                        elms_x_fine.flatten()],
                                                       order=interpolation_order).reshape(
                    elms_x_fine.shape).transpose()

    return F3, coords3


def make_fields(dic_results, sample_location="center", upscale=1, upscale_interpolation_order=1, to_grid=True):
    logger = logging.getLogger()

    settings = dic_results.settings

    if isinstance(settings.mesh.element_def, Q4):
        logger.info("Found Q4 elements")

        if sample_location is "borders":
            seed = 3
            logger.info("Sampling values on a %i x %i grid" % (seed, seed))
            ee, nn = make_grid(seed)

        else:
            logger.info("Sampling the center of the elements")
            ee, nn = make_grid(1)

    elif isinstance(settings.mesh.element_def, BSplineSurface):
        logger.info("Found Spline elements, sampling the center of the elements")
        ee, nn = make_grid_Bspline(settings)
    else:
        raise IOError("No valid element type found")

    F, coords = defgrad_from_nodal_positions(dic_results.xnodesT, dic_results.ynodesT,
                                             settings.mesh,
                                             settings.mesh.element_def, nn,
                                             ee)


    # To make the result formatting consistent across element formulations, we arrange the elements onto a grid
    # with the same dimensions as the mesh. Only supported
    if isinstance(settings.mesh.element_def, Q4) and to_grid:
        if sample_location is "borders":
            F, coords = element_values_to_grid_with_overlaps(F, coords, settings)
        else:
            F, coords = element_values_to_grid(F, coords, settings)

    if upscale != 1:
        logger.info("Upscaling is broken at the moment!")
        F, coords = upsample_mesh(F, coords, upscale, upscale_interpolation_order)

    return Fields(F, coords)


def make_grid(seed):
    if seed == 1:
        return np.meshgrid(np.array([0.5]),
                           np.array([0.5]))
    else:
        return np.meshgrid(np.linspace(0., 1., seed),
                           np.linspace(0., 1., seed))


def make_grid_Bspline(settings):
    # Sample the B-spline element in the center-positions of each sub-element
    shape = (
        settings.mesh.element_def.n_nodes_n, settings.mesh.element_def.n_nodes_e)

    ctrl_e = np.linspace(0., 1.0, shape[0])
    ctrl_n = np.linspace(0., 1.0, shape[1])

    mids_e = (ctrl_e[1:] + ctrl_e[:-1]) / 2.
    mids_n = (ctrl_n[1:] + ctrl_n[:-1]) / 2.

    return np.meshgrid(mids_n, mids_e)


def defgrad_from_nodal_positions(xnodesT, ynodesT, msh, elm, e, n):
    """
    Calculate the deformation gradient from the control point positions
    and the element definitions.

    See the paper for the procedure.

    Parameters
    ----------
    xnodesT : ndarray
        Node position in the x direction
    ynodesT : ndarray
        Node position in the y direction
    msh : Mesh
        A Mesh object
    elm : Element
        A Element object containing the element definitions
    e : ndarray
        The e coordinates of the element
    n : ndarray
        The n coordinates of the element
    """

    # Post Processing
    nEl = msh.n_elms
    ne = np.shape(e)[0]
    nn = np.shape(e)[1]

    # Evaluate shape function gradients on grid within element
    Nn = elm.Nn(e.flatten(), n.flatten())
    dfde = elm.dxNn(e.flatten(), n.flatten())
    dfdn = elm.dyNn(e.flatten(), n.flatten())

    Fstack = []
    coord_stack = []

    for el in range(nEl):
        x_crd = np.einsum('ij,jn -> in', Nn, xnodesT[msh.ele[:, el], :])
        y_crd = np.einsum('ij,jn -> in', Nn, ynodesT[msh.ele[:, el], :])
        dxde = np.einsum('ij,jn -> in', dfde, xnodesT[msh.ele[:, el], :])
        dxdn = np.einsum('ij,jn -> in', dfdn, xnodesT[msh.ele[:, el], :])
        dyde = np.einsum('ij,jn -> in', dfde, ynodesT[msh.ele[:, el], :])
        dydn = np.einsum('ij,jn -> in', dfdn, ynodesT[msh.ele[:, el], :])

        c_confs = np.array([[dxde, dxdn], [dyde, dydn]])
        r_conf_inv = np.linalg.inv(np.rollaxis(c_confs[:, :, :, 0], 2, 0))

        Fs = np.einsum('ijpn,pjk->ikpn', c_confs, r_conf_inv)

        Fs = Fs.reshape((2, 2, ne, nn, -1))

        x_crd = x_crd.reshape((ne, nn, -1))
        y_crd = y_crd.reshape((ne, nn, -1))

        Fstack.append(Fs)

        coord_stack.append(np.array([x_crd, y_crd]))

    # Returns F(nElms, i, j, ide, idn , frame), coords(nElms, i, ide, idn , frame)

    return np.array(Fstack), np.array(coord_stack)


def polar_decomposition(F):
    """
    Perform polar decomposition of F by assuming that F = RU and finding the principal
    values of F^T * F and taking the square root to find U
    :param F:
    :return:
    """
    # Find principal direction directly exploiting that F^t F is symmetric
    G = np.einsum('nji...,njo...->nio...', F, F)
    U = apply_func_in_eigen_space(G, np.sqrt)

    # TODO: Add R to the output
    return U


class Fields(object):
    def __init__(self, F, coords):
        """
        Fields calculates field variables from the DIC-results.
        The implementation is lazy, hence getter methods have to be used.

        NOTE
        ----
        The fields are formatted as follows:

        * Vectors: [elm_id,component_i,element_e_coord,element_n_coord,frame_id]
        * matrices: [elm_id,component_i,component_j,element_e_coord,element_n_coord,frame_id]


        Parameters
        ----------
        dic_results :
            The results from the DIC analysis
        seed : Integer
            The number of grid points which will be evaluated in each direction
        upscale : Float
            Return values on a grid upscale times fines than the original mesh

        Returns
        -------
        A Fields object
        """

        self.logger = logging.getLogger()

        self.__F__ = F
        self.__coords__ = coords
        self.__U__ = polar_decomposition(F)

    def true_strain(self):
        """
        Calculate true strain tensor teps_ij from engineering strain tensor eps_ij as:
        teps_ij = log(eps_ij+1)
        :param eps: Engineering strain tensor eps_ij on the form [nEl,i,j,...]
        :return: True strain tensor teps_ij on the form [nEl,i,j,...]
        """
        return apply_func_in_eigen_space(self.__U__, func=np.log)

    def eng_strain(self):
        return apply_func_in_eigen_space(self.__U__, func=lambda x: x - 1.0)

    def F(self):
        return self.__F__

    def green_strain(self):
        """
        Calculate Green strain tensor from F as G = 0.5*(F^T * F -I)
        :param F: Deformation gradient tensor F_ij on the form [nEl,i,j,...]
        :return: Green Lagrange strain tensor E_ij on the form [nEl,i,j,...]
        """

        # Calculate green strain tensor as 0.5(F^T * F - I)
        return 0.5 * (apply_func_in_eigen_space(self.__U__, func=np.square) - np.eye(2)[np.newaxis, :, :, np.newaxis,
                                                                              np.newaxis, np.newaxis])

    def coords(self):
        return self.__coords__

    def disp(self):
        return self.__coords__[:, :, :, :, :] - self.__coords__[:, :, :, :, 0, np.newaxis]

    def residual(self, frame_id):
        if self.__settings__.store_internals == False:
            raise ValueError("The analysis has to be run with store_internals=True")
        if isinstance(self.__settings__.mesh.element_def, Q4):
            raise NotImplementedError("Q4 residual fields are not yet implemented")
        ref_id = ind_closest_below(frame_id, [ref.image_id for ref in self.__res__.reference])
        ref = self.__res__.reference[ref_id]

        cross_correlation_product = cross_correlation_products(self.__res__.Ic_stack[frame_id], ref.I0_stack)
        self.logger.info("Cross correlation product is %f" % cross_correlation_product)

        return np.abs(self.__res__.Ic_stack[frame_id] - ref.I0_stack)

    def elm_coords(self, frame_id):
        ref_id = ind_closest_below(frame_id, [ref.image_id for ref in self.__res__.reference])
        ref = self.__res__.reference[ref_id]
        return ref.e, ref.n


def apply_func_in_eigen_space(array, func=None):
    """

    Parameters
    ----------
    array
    func

    Returns
    -------

    """
    if func is None:
        func = lambda x: x

    # In order to use the numpy.eig, we need to have the two axes of interest at the end of the array
    E = np.moveaxis(np.moveaxis(array, 1, -1), 1, -1)

    eigvals, eigvecs = np.linalg.eig(E)

    theta = np.arctan(eigvecs[:, :, :, :, 0, 1] / eigvecs[:, :, :, :, 0, 0])

    array_princ = np.zeros_like(array)

    array_princ[:, 0, 0, :, :, :] = func(eigvals[:, :, :, :, 0])
    array_princ[:, 1, 1, :, :, :] = func(eigvals[:, :, :, :, 1])

    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    func_array_temp = np.einsum('ijn...,njo...->noi...', R, array_princ)
    func_array = np.einsum('njo...,ijn...->nio...', func_array_temp, R)
    return func_array


def ind_closest_below(value, list):
    ind = 0
    for i, num in enumerate(list):
        if num < value:
            ind = i

    return ind


def cross_correlation_products(field_a, field_b):
    return np.sum(field_a * field_b) / (
            (np.sum(np.square(field_a)) ** 0.5) * (
            np.sum(np.square(field_b)) ** 0.5))
