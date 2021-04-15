import numpy as np



class Finite_Element(object):
    def __init__(self, order):
        """
        Parent class for finite elements.

        Contains all shape functions and their gradients
        :param order: Element order
        """
        self.fx = self.__fx__(order)
        self.dxfx = self.__dxfx__(order)
        self.dyfx = self.__dyfx__(order)
        self.shape_function_coeff = self.determineCoefficients()

    def __fx__(self, order=1):
        if order == 1:
            n_terms = 4
        elif order == 2:
            n_terms = 8
        elif order == 3:
            n_terms = 16

        def fx_full(x, y):
            return np.array(
                [np.ones(np.size(x)), x, y, x * y, x ** 2., y ** 2., x ** 2. * y, x * y ** 2., x ** 3., y ** 3.,
                 x ** 3. * y,
                 x ** 2. * y ** 2., x * y ** 3., x ** 3. * y ** 2., x ** 2. * y ** 3., x ** 3. * y ** 3.][:n_terms],
                dtype=float).transpose()

        return fx_full

    def __dxfx__(self, order=1):
        if order == 1:
            n_terms = 4
        elif order == 2:
            n_terms = 8
        elif order == 3:
            n_terms = 16

        def dxfx_full(x, y):
            return np.array(
                [np.zeros(np.size(x)), np.ones(np.size(x)), np.zeros(np.size(x)), y, 2. * x, np.zeros(np.size(x)),

                 2. * x * y, y ** 2, 3. * x ** 2, np.zeros(np.size(x)),
                 3. * x ** 2. * y, 2. * x * y ** 2., y ** 3., 3. * x ** 2. * y ** 2., 2. * x * y ** 3.,
                 3. * x ** 2. * y ** 3.][:n_terms], dtype=float).transpose()

        return dxfx_full

    def __dyfx__(self, order=1):
        if order == 1:
            n_terms = 4
        elif order == 2:
            n_terms = 8
        elif order == 3:
            n_terms = 16

        def dyfx_full(x, y):
            return np.array(
                [np.zeros(np.size(x)), np.zeros(np.size(x)), np.ones(np.size(x)), x, np.zeros(np.size(x)), 2. * y,
                 x ** 2., 2. * x * y, np.zeros(np.size(x)), 3. * y ** 2.,
                 x ** 3., 2. * x ** 2. * y, 3. * x * y ** 2., 2. * x ** 3. * y, 3. * x ** 2. * y ** 2.,
                 3. * x ** 3. * y ** 2.][:n_terms], dtype=float).transpose()

        return dyfx_full

    def determineCoefficients(self):
        A = self.fx(self.nodal_xpos, self.nodal_ypos)
        Ainv = np.linalg.inv(A)
        a = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            s = np.zeros(self.n_nodes)
            s[i] = 1.
            a[:, i] = np.dot(Ainv, s)
        return a

    def Nn(self, x, y):
        return np.dot(self.fx(x, y), self.shape_function_coeff)

    def dxNn(self, x, y):
        return np.dot(self.dxfx(x, y), self.shape_function_coeff)

    def dyNn(self, x, y):
        return np.dot(self.dyfx(x, y), self.shape_function_coeff)


class Q4(Finite_Element):
    def __init__(self):
        """
        Quadratic 4-noded Finite Element
        Uses bi-linear interpolation polynomials
        """
        self.nodal_xpos = np.array([0., 1., 1., 0.], dtype=float)

        self.nodal_ypos = np.array([0., 0., 1., 1.], dtype=float)
        self.n_nodes = 4
        self.corner_nodes = np.array([0, 1, 2, 3])
        Finite_Element.__init__(self, 1)