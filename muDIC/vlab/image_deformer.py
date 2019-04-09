import logging
from functools import partial

import numpy as np
import scipy.ndimage as nd
from scipy import optimize


def num_diff(xs, ys, func, component=(1, 1)):
    xs_mapped, ys_mapped = func(xs, ys)
    func_vals_x_grad_x = np.gradient(xs_mapped, axis=1)
    func_vals_x_grad_y = np.gradient(xs_mapped, axis=0)

    func_vals_y_grad_x = np.gradient(ys_mapped, axis=1)
    func_vals_y_grad_y = np.gradient(ys_mapped, axis=0)

    if component == (1, 1):
        return func_vals_x_grad_x  # / xs_spacing_x
    if component == (2, 1):
        return func_vals_x_grad_y
    if component == (1, 2):
        return func_vals_y_grad_x
    if component == (2, 2):
        return func_vals_y_grad_y
    else:
        raise ValueError("Incompatible component request")


def inverse(xs, ys, function, tol=1e-6):
    """Calculates the inverse of a displacement function

    Returns the arguments corresponding to the function values given.
    This is done by a Newton solver which minimizes:

    g(X) = X + f(X,Y) - x = 0
    where x is the coordinate of the displacement material point, having the coordinate X in its undeformed state.
    The coordinate Y is kept constant thoughout the optimization, and the routine is therefore not guarantied to
    converge.

    Parameters
    ----------
    xs : 2d-Numpy array
        x-coordinates
    ys : 2d-Numpy array
        x-coordinates
    function : func
        The displacement function
    tol : float
        The tolerance which the results are checked against

    Returns
    -------
    X,Y
        The coordinates in the undeformed state
    """
    logger = logging.getLogger()

    n_coords = xs.shape

    def f_x(x, y, a_x):
        u_x, _ = function(x, y, img_shape=n_coords)
        return x + u_x - a_x

    def f_y(y, x, a_y):
        _, u_y = function(x, y, img_shape=n_coords)
        return y + u_y - a_y

    x_deformed = xs.flatten()
    x_undeformed_guess = x_deformed.copy()

    y_deformed = ys.flatten()
    y_undeformed_guess = y_deformed.copy()

    x = optimize.newton(f_x, x_undeformed_guess, args=(y_undeformed_guess, x_deformed,))
    y = optimize.newton(f_y, y_undeformed_guess, args=(x_undeformed_guess, y_deformed,))

    error_x = np.max(np.abs(f_x(np.array(x), y_undeformed_guess, x_deformed)))
    error_y = np.max(np.abs(f_y(np.array(y), x_undeformed_guess, y_deformed)))

    if max(error_x, error_y) > tol:
        raise ValueError("The displacement function was not inverted successfully")

    logger.info("Larges error in y is: %f" % error_y)
    logger.info("Larges error in x is: %f" % error_x)

    return np.reshape(x, xs.shape), np.reshape(y, ys.shape)


class ImageDeformer(object):
    def __init__(self, coordinate_mapper, multiplicative,order=4):
        """Image deformer

        Deforms an image according to the settings given at instantiation.


        Parameters
        ----------
        coordinate_mapper : func
            A function taking two coordinates and returns two mapped coordinates
        multiplicative : bool
            Whether the coordinate mapping is multiplicative or not. Use True for deformation gradients
            and use False for displacement fields.
        order : int
            The interpolation order used

        Example
        -------
        Make a ImageDeformed which deforms the image according to a deformation gradient.
        NOTE: You should rather use the factory imageDeformer_from_defGrad.

        >>> import numpy as np
        >>> from muDIC import vlab
        >>> F = np.array([[1.1,.0], [0., 1.0]], dtype=np.float64)
        >>> coordinate_mapper = vlab.image_deformer.CoordinateMapper(partial(map_coords_by_defgrad, F=F))
        >>> vlab.image_deformer.ImageDeformer(coordinate_mapper,multiplicative=True)

        NOTE
        ----
        The image deformed returns a list of deformed images when called, the first being undeformed

        """
        self.multiplicative = multiplicative
        self.coodinate_mapper = coordinate_mapper
        self.order = order

    def __call__(self, img, steps=2):
        n, m = np.shape(img)
        def_imgs = []

        xn, yn = np.meshgrid(np.arange(n), np.arange(m))
        xn_mapped, yn_mapped = xn, yn

        for i in range(steps):
            if i == 0:
                Ic = img
            else:
                if self.multiplicative:
                    xn_mapped, yn_mapped = self.coodinate_mapper(xn_mapped, yn_mapped)
                else:
                    xn_mapped, yn_mapped = self.coodinate_mapper(xn, yn)
                    xn_mapped, yn_mapped = float(i) * (xn_mapped - xn) + xn, float(i) * (
                                yn_mapped - yn) + yn

                Ic = nd.map_coordinates(img, np.array([yn_mapped, xn_mapped]), order=self.order, prefilter=True, cval=0)

            def_imgs.append(Ic)

        return def_imgs


def map_coords_by_defgrad(xs, ys, F):
    """Maps coordinates based on a deformation gradient
    NOTE. The coordinates are first centered, implying that a rotation is about the middle of the image.

    Parameters
    ----------
    xs : 2d-Numpy array
        X-coordinates
    ys : 2d-Numpy array
        Y-coordinates
    F : 2x2-Numpy array
        The deformation gradient

    Returns
    -------
    xs_new,ys_new
        The mapped coordinates in the same ordering as input
    """
    # Center around x and y
    x_mid = np.float(xs.max()) / 2.
    y_mid = np.float(ys.max()) / 2.

    xs_centered = xs - x_mid
    ys_centered = ys - y_mid
    coords = np.array([xs_centered, ys_centered])
    F_inv = np.linalg.inv(F)

    coords_new = np.einsum('ij,jnm->inm', F_inv, coords)
    return coords_new[0, :, :] + x_mid, coords_new[1, :, :] + y_mid


def imageDeformer_from_defGrad(F):
    """Makes a ImageDeformer object from a deformation gradient

    Parameters
    ----------
    F : 2x2-ndarray
        The components of the deformation gradient


    Returns
    -------
    imageDeformer
        An ImageDeformer object which deformes images according to F

    Examples
    --------
    First, we define the deformation gradient and then make an ImageDeformer object:
        >>> import numpy as np
        >>> from muDIC import vlab
        >>> F = np.array([[1.1,.0], [0., 1.0]], dtype=np.float64)
        >>> image_deformer = vlab.imageDeformer_from_defGrad(F)

    """
    return ImageDeformer(partial(map_coords_by_defgrad, F=F), multiplicative=True)


def imageDeformer_from_uFunc(uFunc, **kwargs):
    """Makes a ImageDeformer object from a displacement function

    Parameters
    ----------
    uFunc : func
        The displacement function

    Returns
    -------
    imageDeformer
        An ImageDeformer object which deformes images according to the displacement function

    Examples
    --------
    First, we define the deformation gradient and then make an ImageDeformer object:
        >>> from muDIC import vlab
        >>> image_deformer = vlab.imageDeformer_from_uFunc(vlab.deformation_fields.bilat_harmonic,omega=4*np.pi)

    """
    coordinate_generator = partial(uFunc, **kwargs)
    coordinate_mapper_inv = partial(inverse, function=coordinate_generator)
    return ImageDeformer(coordinate_mapper_inv, multiplicative=False)
