import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from muDIC.elements.b_splines import BSplineSurface
from muDIC.elements.q4 import Q4


class Fields(object):
    # TODO: Remove Q4 argument. This should be detected automaticaly
    def __init__(self, dic_results, seed=21, upscale=1, interpolation_order=1):
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

        # The type is implicitly checked by using the interface
        self.__res__ = dic_results
        self.__settings__ = dic_results.settings
        self.interpolation_order = interpolation_order

        if isinstance(self.__settings__.mesh.element_def, Q4):
            q4 = True
            seed = 1
            self.logger.info("Post processing results from Q4 elements. The seed variable is ignored and the values "
                             "are extracted at the element centers. Use the upscale value to get interpolated fields.")
        else:
            q4 = False
            self.logger.info("Post processing results from B-spline elements. The upscale variable is ignored. Use "
                             "the seed varialbe to set the number of gridpoints to be evaluated along each element "
                             "axis.")

        self.__ee__, self.__nn__ = self.__generate_grid__(seed)

        self.__F__, self.__coords__ = self._deformation_gradient_(self.__res__.xnodesT, self.__res__.ynodesT,
                                                                  self.__settings__.mesh,
                                                                  self.__settings__.mesh.element_def, self.__nn__,
                                                                  self.__ee__)

        # To make the result formatting consistent across element formulations, we arrange the elements onto a grid
        # with the same dimensions as the mesh. If up-scaling is used, we determine the values between element centers
        # by using 3rd order spline interpolation.
        n_frames = self.__F__.shape[-1]

        if q4:
            # Flatten things form multiple elements to a grid of elements
            grid_shape = (self.__settings__.mesh.n_ely, self.__settings__.mesh.n_elx)
            n_frames = self.__F__.shape[-1]
            self.__F2__ = np.zeros(
                (1, 2, 2, self.__settings__.mesh.n_elx, self.__settings__.mesh.n_ely, self.__F__.shape[-1]))
            for i in range(2):
                for j in range(2):
                    for t in range(n_frames):
                        self.__F2__[0, i, j, :, :, t] = self.__F__[:, i, j, 0, 0, t].reshape(grid_shape).transpose()

            self.__coords2__ = np.zeros(
                (1, 2, self.__settings__.mesh.n_elx, self.__settings__.mesh.n_ely, self.__F__.shape[-1]))
            for i in range(2):
                for t in range(n_frames):
                    self.__coords2__[0, i, :, :, t] = self.__coords__[:, i, 0, 0, t].reshape(grid_shape).transpose()

            # Overwrite the old results
            # TODO: Remove overwriting results as this is a painfully non-functional thing to do...
            self.__coords__ = self.__coords2__
            self.__F__ = self.__F2__

            self.__coords__ = self.__coords2__
            self.__F__ = self.__F2__

        if upscale != 1.:
            elms_y_fine, elms_x_fine = np.meshgrid(np.arange(0, self.__coords__.shape[-3]-1, 1. / upscale),
                                                   np.arange(0, self.__coords__.shape[-2]-1 , 1. / upscale))

            self.__F3__ = np.zeros(
                (1, 2, 2, elms_x_fine.shape[1], elms_x_fine.shape[0], self.__F__.shape[-1]))

            self.__coords3__ = np.zeros(
                (1, 2, elms_x_fine.shape[1], elms_x_fine.shape[0], self.__F__.shape[-1]))

            for i in range(2):
                for t in range(n_frames):
                    self.__coords3__[0, i, :, :, t] = map_coordinates(self.__coords__[0, i, :, :, t],
                                                                      [elms_y_fine.flatten(),
                                                                       elms_x_fine.flatten()],
                                                                      order=self.interpolation_order).reshape(
                        elms_x_fine.shape).transpose()

            for i in range(2):
                for j in range(2):
                    for t in range(n_frames):
                        self.__F3__[0, i, j, :, :, t] = map_coordinates(self.__F__[0, i, j, :, :, t],
                                                                        [elms_y_fine.flatten(),
                                                                         elms_x_fine.flatten()],
                                                                        order=self.interpolation_order).reshape(
                            elms_x_fine.shape).transpose()

            self.__coords__ = self.__coords3__
            self.__F__ = self.__F3__

    def __generate_grid__(self, seed):

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
                    self.__res__.settings.mesh.element_def.n_nodes_n, self.__res__.settings.mesh.element_def.n_nodes_e)

                ctrl_e = np.linspace(0., 1.0, shape[0])
                ctrl_n = np.linspace(0., 1.0, shape[1])

                mids_e = (ctrl_e[1:] + ctrl_e[:-1]) / 2.
                mids_n = (ctrl_n[1:] + ctrl_n[:-1]) / 2.

                return np.meshgrid(mids_n, mids_e)

    @staticmethod
    def _deformation_gradient_(xnodesT, ynodesT, msh, elm, e, n):
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

    @staticmethod
    def _green_deformation_(F):
        """
        Calculate Green deformation tensor from deformation as G = F^T * F
        :param F:
        :return:
        """
        E11 = F[:, 0, 0, :, :, :] ** 2. + F[:, 1, 0, :, :, :] ** 2.

        E12 = F[:, 0, 0, :, :, :] * F[:, 1, 0, :, :, :] + F[:, 0, 1, :, :, :] * F[:, 1, 1, :, :, :]

        E22 = F[:, 0, 1, :, :, :] ** 2. + F[:, 1, 1, :, :, :] ** 2.

        E = np.array([[E11, E12], [E12, E22]])

        E[E == np.nan] = 0.

        return np.moveaxis(E, 2, 0)

    @staticmethod
    def _green_strain_(F):
        """
        Calculate Green strain tensor from F as G = 0.5*(F^T * F -I)
        :param F: Deformation gradient tensor F_ij on the form [nEl,i,j,...]
        :return: Green Lagrange strain tensor E_ij on the form [nEl,i,j,...]
        """

        Green_deformation = np.einsum('nji...,njo...->nio...', F, F)

        I = np.eye(2, dtype=np.float)

        # Calculate green strain tensor as 0.5(F^T * F - I)
        Green_strain = 0.5 * (Green_deformation - I[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis])

        return Green_strain

    @staticmethod
    def _principal_strain_(G):


        E_temp = np.moveaxis(G, 1, -1)
        E = np.moveaxis(E_temp, 1, -1)

        eigvals, eigvecs = np.linalg.eig(E)



        ld1 = np.sqrt(eigvals[:, :, :, :, 0])
        ld2 = np.sqrt(eigvals[:, :, :, :, 1])

        ev1 = eigvecs[:, :, :, :, 0, 0]
        ev2 = eigvecs[:, :, :, :, 0, 1]


        ld = np.moveaxis(np.array([ld1, ld2]), 0, 1)
        ev = np.moveaxis(np.array([ev1, ev2]), 0, 1)


        return ld, ev

    @staticmethod
    def _polar_decomposition_(F):
        """
        Perform polar decomposition of F by assuming that F = RU and finding the principal
        values of F^T * F and taking the square root to find U
        :param F:
        :return:
        """
        # Find principal direction directly exploiting that F^t F is symmetric
        G = np.einsum('nji...,njo...->nio...', F, F)
        tan_2_theta = 2.0 * G[:, 0, 1, :, :, :] / (G[:, 0, 0, :, :, :] - G[:, 1, 1, :, :, :])
        theta = np.arctan(tan_2_theta) / 2.
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        G_temp = np.einsum('ijn...,njo...->noi...', R, G)
        G_principal = np.einsum('njo...,ijn...->nio...', G_temp, R)

        # Now, we can find the principal stretches by taking the square root
        G_principal[G_principal < 0.] = 0.
        U = np.sqrt(G_principal)

        return R, U


    @staticmethod
    def _engineering_strain_(E):
        """
        Calculate engineering strain from Green Lagrange strain tensor E_ij as:
        eps_ii = sqrt(1+E_ii)-1 and
        gamma_ij = 2E_ij/sqrt((1+E_ii)*(1+E_jj))
        :param E: Green Lagrange strain tensor E_ij on the form [nEl,i,j,...]
        :return: Engineering strain tensor eps_ij on the form [nEl,i,j,...]
        """
    #    # Find the stretch tensor
        # First, the Green deformation tensor "C²"
        logger = logging.getLogger()
        G = 2.*E + np.eye(2)[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        n_frames = G.shape[-1]
        eps = np.zeros_like(G)
        for i in range(n_frames):
            #G = G[:,:,:,:,:,1:]
            try:
                tan_2_theta = 2.0 * G[:,0,1,:,:,i]/(G[:,0,0,:,:,i]-G[:,1,1,:,:,i])
            except FloatingPointError:
                tan_2_theta = np.zeros_like(G[:,0,1,:,:,i])
            theta = np.arctan(tan_2_theta) / 2.


            R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            G_temp = np.einsum('ijn...,njo...->noi...', R, G[:,:,:,:,:,i])

            G_principal = np.einsum('njo...,ijn...->nio...', G_temp, R)

            G_principal[G_principal < 0.] = 0.
            eps_princ = np.sqrt(G_principal) - np.eye(2)[np.newaxis, :, :, np.newaxis, np.newaxis]

            theta = -np.arctan(tan_2_theta) / 2.
            R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            eps_temp = np.einsum('ijn...,njo...->noi...', R, eps_princ)
            eps[:,:,:,:,:,i] = np.einsum('njo...,ijn...->nio...', eps_temp, R)
        return eps




    @staticmethod
    def _true_strain_(eps):
        """
        Calculate true strain tensor teps_ij from engineering strain tensor eps_ij as:
        teps_ij = log(eps_ij+1)
        :param eps: Engineering strain tensor eps_ij on the form [nEl,i,j,...]
        :return: True strain tensor teps_ij on the form [nEl,i,j,...]
        """
        return np.log(eps + 1.)

    def true_strain(self):
        E = self._green_strain_(self.__F__)
        engineering_strains = self._engineering_strain_(E)
        return self._true_strain_(engineering_strains)

    def eng_strain(self):
        E = self._green_strain_(self.__F__)
        return self._engineering_strain_(E)

    def F(self):
        return self.__F__

    def green_strain(self):
        return self._green_strain_(self.__F__)

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


class Visualizer(object):
    def __init__(self, fields, images=False):
        """
        Visualizer for field variables.

        Parameters
        ----------
        fields : Fields object
            The Fields object contains all the variables that can be plotted.
        images : ImageStack object
            The stack of images corresponding to Fields

        Returns
        -------
        A Visualizer Object
        """
        if isinstance(fields, Fields):
            self.fields = fields
        else:
            raise ValueError("Only instances of Fields are accepted")

        self.images = images
        self.logger = logging.getLogger()

    def show(self, field="displacement", component=(0, 0), frame=0, quiverdisp=False, **kwargs):
        """
        Show the field variable

        Parameters
        ----------
        field : string
            The name of the field to be shown. Valid inputs are:
                "true strain"
                "eng strain"
                "disp"
                "green strain"
                "residual"

        component : tuple with length 2
            The components of the fields. Ex. (0,1).
            In the case of vector fields, only the first index is used.
        frame : Integer
            The frame number of the field

        """

        keyword = field.replace(" ", "").lower()

        if keyword == "truestrain":
            fvar = self.fields.true_strain()[0, component[0], component[1], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]

        elif keyword in ("F", "degrad", "deformationgradient"):
            fvar = self.fields.F()[0, component[0], component[1], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]

        elif keyword == "engstrain":
            fvar = self.fields.eng_strain()[0, component[0], component[1], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]

        elif keyword in ("displacement", "disp", "u"):
            fvar = self.fields.disp()[0, component[0], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]

        elif keyword in ("coordinates", "coords", "coord"):
            fvar = self.fields.coords()[0, component[0], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]


        elif keyword == "greenstrain":
            fvar = self.fields.green_strain()[0, component[0], component[1], :, :, frame]
            xs, ys = self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame]

        elif keyword == "residual":
            fvar = self.fields.residual(frame)
            xs, ys = self.fields.elm_coords(frame)

        else:
            self.logger.info("No valid field name was specified")
            return

        if np.ndim(fvar) == 2:
            if self.images:
                n, m = self.images[frame].shape
                plt.imshow(self.images[frame], cmap=plt.cm.gray, origin="lower", extent=(0, m, 0, n))

            if quiverdisp:
                plt.quiver(self.fields.coords()[0, 0, :, :, frame], self.fields.coords()[0, 1, :, :, frame],
                           self.fields.disp()[0, 0, :, :, frame], self.fields.disp()[0, 1, :, :, frame], **kwargs)
            else:
                plt.contourf(xs, ys, fvar, 50, **kwargs)
                plt.colorbar()

        elif np.ndim(fvar) == 1:
            plt.tricontourf(xs,ys,fvar,**kwargs)
            plt.colorbar()

        plt.show()


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
