import logging

import numpy as np
from numba import jit, prange

from .fieldInterpolator import FieldInterpolator


# TODO: This thing should be refactored such that it runs i parallel

class BSplineSurface(FieldInterpolator):
    def __init__(self, deg_e=3, deg_n=3, n_ctrlpts_e=6, n_ctrlpts_n=6):
        """
        B-spline surface

        This B-spline surface is based on algorithms found in:

        >>> "The NURBS book" by Les Piegl and Wayne Tiller ISBN: 9783540615453

        The definition is made such that the Basis-functions are evaluated first allowing for evaluation of the
        surface by a simple dot product of the basis functions and the control point coordinates without recalculating
        the basis functions.


         Parameters
         ----------
         deg_e : int
             The order of the basis function in the e-direction
         deg_n : int
             The order of the basis function in the n-direction
         n_ctrlpts_e : int
             The number of control points in the e-direction
         n_ctrlpts_n : int
             The number of control points in the n-direction
         Returns
         -------
         B-spline surface object


         Examples
         --------
            >>> import muDIC as dic
            >>> import numpy as np
            >>> es,ns = np.meshgrid(np.linspace(0.,1.,10),np.linspace(0.,1.,10))
            >>> surface = dic.elements.B_spline_surface(deg_e=3, deg_n=3, n_ctrlpts_e=6, n_ctrlpts_n=6)
            >>> bases = surface.Nn(es.flatten(),ns.flatten())
         """

        self.logger = logging.getLogger()

        if isinstance(deg_e, int) and isinstance(deg_n, int):
            self.degree_e = deg_e
            self.degree_n = deg_n

        if isinstance(n_ctrlpts_e, int) and isinstance(n_ctrlpts_n, int):
            self.n_nodes_e = np.int64(n_ctrlpts_e)
            self.n_nodes_n = np.int64(n_ctrlpts_n)

        self.__gen_uniform_knotvectors__()

        self._indices_ = self._make_ctrlpt_indices_()
        self.ctrl_pos_e, self.ctrl_pos_n = self._make_grid_()
        self.n_nodes = self.n_nodes_e * self.n_nodes_n

    def _update_internals_(self):
        self.__gen_uniform_knotvectors__()
        self._indices_ = self._make_ctrlpt_indices_()
        self.ctrl_pos_e, self.ctrl_pos_n = self._make_grid_()
        self.n_nodes = self.n_nodes_e * self.n_nodes_n

    def _make_ctrlpt_indices_(self):
        return np.arange(self.n_nodes_e * self.n_nodes_n).reshape((self.n_nodes_e, self.n_nodes_n))

    def set_n_nodes(self, grid_shape):
        n_nodes_e, n_nodes_n = grid_shape

        if n_nodes_e < self.degree_e + 1 or n_nodes_e < self.degree_e + 1:
            raise ValueError("The number of nodes along an axis has to be larger than the degree of the polynomial +1")

        self.n_nodes_e = np.int64(n_nodes_e)
        self.n_nodes_n = np.int64(n_nodes_n)

        self._update_internals_()

    def set_degree(self, degs):
        dege, degn = degs
        self.degree_e = np.int64(dege)
        self.degree_n = np.int64(degn)

        self._update_internals_()

    def Nn(self, es, ns):
        """
         Evaluate the B-spline surface basis functions at the given coordinates in [0,1].
         This algorithm corresponds to A3.5 but without the multiplication with the nodal coordinates.

         >>> "The NURBS book" by Les Piegl and Wayne Tiller ISBN: 9783540615453

        The order to evaluate the the surface for a given set of control points, np.dot(Nn(e,n), ctrlpts)
        has to be evaluated.


          Parameters
          ----------
          es : ndarray
              1d array, with the coordinates to be evaluated the e-direction
          ns : ndarray
              1d array, with the coordinates to be evaluated the n-direction

          Returns
          -------
          basis_funcs : ndarray
                2D array, with the evaluated basis functions.
                The shape of the matrix is size(es) x num_crtl_pts

        """

        if es.ndim != 1 or ns.ndim != 1:
            raise ValueError("Only 1d arrays are accepted")

        if ns.max() > 1. or ns.min() < 0. or es.max() > 1. or es.min() < 0.:
            raise ValueError("B-spline coordinates has to be within [0,1]")

        spans_e = self._find_span_array_(self.degree_e, self._knotvector_e_, self.n_nodes_e, ns)

        bases_e = self.basis_functions_array(self.degree_e, self._knotvector_e_, spans_e, ns)

        spans_n = self._find_span_array_(self.degree_n, self._knotvector_n_, self.n_nodes_n, es)

        bases_n = self.basis_functions_array(self.degree_n, self._knotvector_n_, spans_n, es)

        results = self.__partial_outer_product__(self.n_nodes_e, self.n_nodes_n, len(es), spans_n, bases_n, spans_e,
                                                 bases_e,
                                                 self.degree_n,
                                                 self.degree_e, self._indices_)

        return results

    def dxNn(self, vs, us):
        """
         Evaluate the B-spline surface basis function derivatives in X at the given coordinates in [0,1].
         This algorithm corresponds to A3.5 but without the multiplication with the nodal coordinates.

         >>> "The NURBS book" by Les Piegl and Wayne Tiller ISBN: 9783540615453

        The order to evaluate the the surface for a given set of control points, np.dot(Nn(e,n), ctrlpts)
        has to be evaluated.


          Parameters
          ----------
          es : ndarray
              1d array, with the coordinates to be evaluated the e-direction
          ns : ndarray
              1d array, with the coordinates to be evaluated the n-direction

          Returns
          -------
          basis_funcs : ndarray
                2D array, with the evaluated basis function derivatives.
                The shape of the matrix is size(es) x num_crtl_pts

        """
        return self.__dKNn__(vs, us, 0, 1)

    def dyNn(self, vs, us):
        """
         Evaluate the B-spline surface basis function derivatives in Y at the given coordinates in [0,1].
         This algorithm corresponds to A3.5 but without the multiplication with the nodal coordinates.

         >>> "The NURBS book" by Les Piegl and Wayne Tiller ISBN: 9783540615453

        The order to evaluate the the surface for a given set of control points, np.dot(Nn(e,n), ctrlpts)
        has to be evaluated.


          Parameters
          ----------
          es : ndarray
              1d array, with the coordinates to be evaluated the e-direction
          ns : ndarray
              1d array, with the coordinates to be evaluated the n-direction

          Returns
          -------
          basis_funcs : ndarray
                2D array, with the evaluated basis function derivatives.
                The shape of the matrix is size(es) x num_crtl_pts

        """

        return self.__dKNn__(vs, us, 1, 0)

    def __dKNn__(self, vs, us, k, l):

        du = 1
        dv = 1

        n_pts = np.min([np.size(us), np.size(vs)]).astype(np.int64)

        spans_e = self._find_span_array_(self.degree_e, self._knotvector_e_, self.n_nodes_e, us)

        bases_ders_e = self.__basis_functions_ders_array__(self.degree_e, self._knotvector_e_, spans_e, us, du)

        span_n = self._find_span_array_(self.degree_n, self._knotvector_n_, self.n_nodes_n, vs)

        bases_ders_n = self.__basis_functions_ders_array__(self.degree_n, self._knotvector_n_, span_n, vs, dv)

        results_ordered = self.__partial_outer_product__(self.n_nodes_n, self.n_nodes_e, n_pts, span_n,
                                                         bases_ders_n[:, l, :],
                                                         spans_e,
                                                         bases_ders_e[:, k, :], self.degree_n, self.degree_e,
                                                         self._indices_)

        return results_ordered

    def _make_grid_(self):

        es = np.linspace(0., 1., self.n_nodes_e)
        ns = np.linspace(0., 1., self.n_nodes_n)

        ctrlpts_epos, ctrlpts_npos = np.meshgrid(ns, es)

        return ctrlpts_epos.flatten(), ctrlpts_npos.flatten()

    def __gen_uniform_knotvectors__(self):
        self._knotvector_e_ = np.array(self._gen_uniform_knotvector_(self.degree_e, self.n_nodes_e))
        self._knotvector_n_ = np.array(self._gen_uniform_knotvector_(self.degree_n, self.n_nodes_n))

    @staticmethod
    @jit(nopython=True)
    def _find_span_(degree, knotvector, num_ctrlpts, knot, tol=1e-5):

        """ Algorithm A2.1 of The NURBS Book by Piegl & Tiller."""
        # Number of knots; m + 1
        # Number of control points; n + 1
        # n = m - p - 1; where p = degree
        # m = len(knotvector) - 1
        # n = m - degree - 1
        n = num_ctrlpts - 1
        if np.abs(knotvector[n + 1] - knot) <= tol:
            return int(n)

        low = degree
        high = n + 1
        mid = int((low + high) / 2)

        while (knot < knotvector[mid]) or (knot >= knotvector[mid + 1]):
            if knot < knotvector[mid]:
                high = mid
            else:
                low = mid
            mid = int((low + high) / 2)

        return int(mid)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _find_span_array_(degree, knotvector, num_ctrlpts, knots, tol=1e-5):

        results = np.zeros(len(knots), dtype=np.int64)

        # These are knot invariant
        n = num_ctrlpts - 1
        n_knots = len(knots)

        for i in prange(n_knots):
            knot = knots[i]

            """ Algorithm A2.1 found in "The NURBS Book" by Piegl & Tiller."""
            if np.abs(knotvector[n + 1] - knot) <= tol:
                results[i] = np.int64(n)

            else:
                low = degree
                high = n + 1
                mid = int((low + high) / 2)
                while (knot < knotvector[mid]) or (knot >= knotvector[mid + 1]):
                    if knot < knotvector[mid]:
                        high = mid
                    else:
                        low = mid
                    mid = int((low + high) / 2)

                results[i] = np.int64(mid)
        return results

    @staticmethod
    @jit(nopython=True)
    def __basis_functions_ders_array__(degree=0, knotvector=(), spans=0, knots=0, order=0):
        """ Algorithm A2.3 of The NURBS Book by Piegl & Tiller."""
        # Initialize variables for easy access
        left = np.zeros(degree + 1)
        right = np.zeros(degree + 1)
        # ndu = [[None for x in range(degree+1)] for y in range(degree+1)]
        ndu = np.zeros((degree + 1, degree + 1))

        # N[0][0] = 1.0 by definition

        # ders = np.zeros(((min(degree, order) + 1), degree + 1))
        n_knots = len(knots)
        size = min(degree, order) + 1
        results = np.zeros((n_knots, size, degree + 1))

        counter = 0
        # for knot in knots:
        for counter in range(len(knots)):
            ndu[0, 0] = 1.0
            span = spans[counter]
            knot = knots[counter]

            j = int(0)
            for j in range(1, degree + 1):
                left[j] = knot - knotvector[span + 1 - j]
                right[j] = knotvector[span + j] - knot
                saved = 0.0
                r = 0
                for r in range(r, j):
                    # Lower triangle
                    ndu[j, int(r)] = right[int(r) + 1] + left[int(j) - r]
                    temp = ndu[r, j - 1] / ndu[j, r]
                    # Upper triangle
                    ndu[r][j] = saved + (right[r + 1] * temp)
                    saved = left[j - r] * temp
                ndu[j, j] = saved

            # Load the basis functions
            ders = np.zeros(((min(degree, order) + 1), degree + 1))
            for j in range(0, degree + 1):
                ders[0, j] = ndu[j, degree]

            # Start calculating derivatives
            a = np.zeros((degree + 1, 2))
            # Loop over function index
            for r in range(0, degree + 1):
                # Alternate rows in array a
                s1 = 0
                s2 = 1
                a[0, 0] = 1.0
                # Loop to compute k-th derivative
                for k in range(1, order + 1):
                    d = 0.0
                    rk = r - k
                    pk = degree - k
                    if r >= k:
                        a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                        d = a[s2, 0] * ndu[rk, pk]
                    if rk >= -1:
                        j1 = 1
                    else:
                        j1 = -rk
                    if (r - 1) <= pk:
                        j2 = k - 1
                    else:
                        j2 = degree - r
                    for j in range(j1, j2 + 1):
                        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                        d += (a[s2, j] * ndu[rk + j, pk])
                    if r <= pk:
                        a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                        d += (a[s2, k] * ndu[r, pk])
                    ders[k, int(r)] = d

                    # Switch rows
                    j = s1
                    s1 = s2
                    s2 = j

            # Multiply through by the the correct factors
            r_fl = float(degree)
            for k in range(1, order + 1):
                for j in range(0, degree + 1):
                    ders[k, j] *= r_fl
                r_fl *= (degree - k)

            # Return the basis function derivatives list
            results[counter] = ders

        return results

    @staticmethod
    def __partial_outer_product__(n_nodes_y, n_nodes_x, n_pts, span_vs, bases_vs, span_us, bases_us, degree_v, degree_u,
                                  indices):
        # Declare results array

        results = np.zeros(n_pts * n_nodes_x * n_nodes_y, dtype=np.float64)

        # Calculate node span in x and y direction
        deg_range_u = np.array(range(0, degree_u + 1), dtype=np.int)
        deg_range_v = np.array(range(0, degree_v + 1), dtype=np.int)

        # Generate index matrix for all points
        u_ind = (span_us[:, np.newaxis] - degree_u + deg_range_u[np.newaxis, :])[:, np.newaxis, :]
        v_ind = (span_vs[:, np.newaxis] - degree_v + deg_range_v[np.newaxis, :])[:, :, np.newaxis]

        # Determine pixel indices in global frame
        index_matrix = indices[u_ind, v_ind]

        # Allow for flattening of index array
        index_matrix += (np.arange(n_pts) * n_nodes_x * n_nodes_y)[:, np.newaxis, np.newaxis]

        index_matrix_ordered = np.moveaxis(index_matrix, 0, -1)

        # Generate outer product of the derivatives vectors for a stack of vectors
        big_outer = np.einsum('ij,il->jli', bases_vs, bases_us)

        results[index_matrix_ordered] = big_outer

        return results.reshape((n_pts, -1))

    @staticmethod
    def _gen_uniform_knotvector_(degree=0, num_ctrlpts=0):
        """ Generates a uniformly-spaced knot vector using the degree and the number of control points.
        :param degree: degree of the knot vector direction
        :type degree: integer
        :param num_ctrlpts: number of control points on that direction
        :type num_ctrlpts: integer
        :return: knot vector
        :rtype: list
        """
        if degree == 0 or num_ctrlpts == 0:
            raise ValueError("Input values should be different than zero.")

        # Min and max knot vector values
        knot_min = 0.0
        knot_max = 1.0

        # Equation to use: m = n + p + 1
        # p: degree, n+1: number of ctrlpts; m+1: number of knots
        m = degree + num_ctrlpts + 1

        # Initialize return value and counter
        knotvector = []
        i = 0

        # First degree+1 knots are "knot_min"
        while i < degree + 1:
            knotvector.append(knot_min)
            i += 1

        # Calculate a uniform interval for middle knots
        num_segments = (m - (degree + 1) * 2) + 1  # number of segments in the middle
        spacing = (knot_max - knot_min) / num_segments  # spacing between the knots (uniform)
        midknot = knot_min + spacing  # first middle knot
        # Middle knots
        while i < m - (degree + 1):
            knotvector.append(midknot)
            midknot += spacing
            i += 1

        # Last degree+1 knots are "knot_max"
        while i < m:
            knotvector.append(knot_max)
            i += 1

        # Return autogenerated knot vector
        return knotvector

    @staticmethod
    @jit(nopython=True, parallel=False)
    def basis_functions_array(degree=0, knotvector=(), spans=0, knots=0):
        num_knots = len(knots)
        results = np.zeros((num_knots, degree + 1), dtype=np.float64)

        left = np.zeros(degree + 1, dtype=np.float64)
        right = np.zeros(degree + 1, dtype=np.float64)
        N = np.zeros(degree + 1, dtype=np.float64)

        for i in prange(num_knots):
            knot = knots[i]
            span = spans[i]

            """ Algorithm A2.2 of The NURBS Book by Piegl & Tiller."""

            left[:] = 0.0
            right[:] = 0.0
            N[:] = 0.0

            # N[0] = 1.0 by definition
            N[0] = 1.0

            for j in range(1, degree + 1):
                left[j] = knot - knotvector[span + 1 - j]
                right[j] = knotvector[span + j] - knot
                saved = np.float64(0.0)
                for r in range(0, j):
                    temp = N[r] / (right[r + 1] + left[j - r])
                    N[r] = saved + right[r + 1] * temp
                    saved = left[j - r] * temp
                N[j] = saved
                results[i, :] = N[:]
        return results
