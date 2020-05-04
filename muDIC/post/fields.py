import logging

import numpy as np
from scipy.ndimage import map_coordinates

from muDIC.elements import Q4


def makeFields(dic_results, seed=21, upscale=1, interpolation_order=1):
    logger = logging.getLogger()

    # The type is implicitly checked by using the interface
    res = dic_results
    settings = dic_results.settings
    interpolation_order = interpolation_order

    if isinstance(settings.mesh.element_def, Q4):
        q4 = True
        seed = 1
        logger.info("Post processing results from Q4 elements. The seed variable is ignored and the values "
                    "are extracted at the element centers. Use the upscale value to get interpolated fields.")
    else:
        q4 = False
        logger.info("Post processing results from B-spline elements. The upscale variable is ignored. Use "
                    "the seed varialbe to set the number of gridpoints to be evaluated along each element "
                    "axis.")

    ee, nn = make_grid(seed, settings)

    F, coords = defgrad_from_nodal_positions(res.xnodesT, res.ynodesT,
                                             settings.mesh,
                                             settings.mesh.element_def, nn,
                                             ee)

    # To make the result formatting consistent across element formulations, we arrange the elements onto a grid
    # with the same dimensions as the mesh. If up-scaling is used, we determine the values between element centers
    # by using 3rd order spline interpolation.
    n_frames = F.shape[-1]

    if q4:
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

        # Overwrite the old results
        # TODO: Remove overwriting results as this is a painfully non-functional thing to do...

        coords = coords2
        F = F2

    if upscale != 1.:
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

        coords = coords3
        F = F3

    return Fields(F, coords)


def make_grid(seed, settings):
    # TODO: Remove hack:
    if seed == 1:
        return np.meshgrid(np.array([0.5]),
                           np.array([0.5]))

    else:

        if np.ndim(seed) == 1:
            return np.meshgrid(np.linspace(0., 1., seed[0]),
                               np.linspace(0., 1., seed[1]))

        else:

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
    def __init__(self, F,coords):
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
        return 0.5 * (apply_func_in_eigen_space(self.__U__,func=np.square) - np.eye(2)[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis])

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