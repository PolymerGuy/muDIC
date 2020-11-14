from __future__ import print_function

import logging
from copy import copy

import numpy as np

from ..IO.image_stack import ImageStack
from ..elements.b_splines import BSplineSurface
from ..elements.q4 import Q4


def scale_to_unit(array):
    return (array - array.min()) / (array.max() - array.min())


# Abaqus reader prototype
def mesh_from_abaqus(inpfile_name, unit_dim=False):
    """Generate mesh from Abaqus inpt file

     The nodal positions and nodal connectivities are extracted from the input file.


     Example
     -------
     Let's import the mesh provided in examples and plot the nodal positions

     >>> from muDIC import mesh_from_abaqus
     >>> import matplotlib.pyplot as plt
     >>> nodes_x,nodes_y,con_mat = mesh_from_abaqus('./abaqusMeshes/ring.inp')
     >>> plt.plot(nodes_x,nodes_y)
     >>> plt.show()


     Note
     ----
     The supported Abaqus mesh types are:
         * 4-Noded planar


     Parameters
     ----------
     inpfile_name : string
         Name of Abaqus input file containing the mesh
     unit_dim : bool
         Scale the mesh such that it spans form zero to one in both directions
     """

    with open(inpfile_name, 'r') as f:
        nodal = []
        connectivity = []

        read_nodes = False
        read_elm = False
        last_ln = ''
        for line in f:
            ln = line.strip()

            if ln == '*Node' and '*Part' in last_ln:
                read_nodes = True
            elif '*Element' in ln:
                read_nodes = False
                read_elm = True

            if '*End' in ln:
                read_elm = False

            if read_nodes:
                nodal.append(ln)
            if read_elm:
                connectivity.append(ln)

            last_ln = ln

    # Convert the strings to numerical data
    nodes = np.array([ind.split(',') for ind in nodal[1:]], dtype=np.float)

    # Remove the node labels
    nodes = nodes[:, 1:]

    con_mat = np.array([elm.split(',') for elm in connectivity[1:]], dtype=np.int)

    # The node indices are to be zero indexed, remove labels
    con_mat = con_mat[:, 1:] - 1

    xnodes = nodes[:, 0]
    ynodes = nodes[:, 1]
    con_mat = con_mat.transpose()

    if len(xnodes) != len(ynodes) or con_mat.shape[0]!=4:
        raise IOError("Invalid Abaqus input file")

    if unit_dim:
        xnodes = scale_to_unit(xnodes)
        ynodes = scale_to_unit(ynodes)

    return Mesh(Q4(), xnodes, ynodes, con_mat)


def make_grid_Q4(c1x, c1y, c2x, c2y, nx, ny, elm):
    # type: (float, float, float, float, int, int, instance) -> object
    """
    Makes regular grid for the given corned coordinates, number of elements along each axis and finite element definitions
    :rtype: np.array,np.array,np.array
    :param c1x: X-position of upper left corner
    :param c1y: Y-position of upper left corner
    :param c2x: X-position of lower right corner
    :param c2y: Y-position of lower right corner
    :param nx:  Number of elements along X-axis
    :param ny:  Number of elements along Y-axis
    :param elm: Finite element instance
    :return: Connectivity matrix, X-coordinates of nodes, Y-Coordinates of nodes
    """

    n_decimals = 2

    elmwidth = float(c2x - c1x) / float(nx)
    elmheigt = float(c2y - c1y) / float(ny)

    xnodes = elm.nodal_xpos * elmwidth
    ynodes = elm.nodal_ypos * elmheigt

    elements = []
    nodes = set()

    for i in range(ny):
        for j in range(nx):
            elements.append(
                zip(np.around(ynodes[:] + elmheigt * i, n_decimals), np.around(xnodes[:] + elmwidth * j, n_decimals)))
            nodes.update(
                zip(np.around(ynodes[:] + elmheigt * i, n_decimals), np.around(xnodes[:] + elmwidth * j, n_decimals)))

    nodes = sorted(list(nodes))

    con_matrix = []

    for e in range(nx * ny):
        con_matrix.append(list(map(nodes.index, list(elements[e]))))

    ynod, xnod = zip(*nodes)
    ynode = np.array(ynod) + c1y
    xnode = np.array(xnod) + c1x

    return np.array(con_matrix).transpose(), xnode, ynode


def make_grid(c1x, c1y, c2x, c2y, ny, nx, elm):
    """
    Makes regular grid for the given corner coordinates, number of elements along each axis and finite element
    definitions.




     Parameters
     ----------
    c1x : float
        X-position of upper left corner
    c1y : float
        Y-position of upper left corner
    c2x : float
        X-position of lower right corner
    c2y : float
        Y-position of lower right corner
    nx : int
        Number of elements along X-axis
    ny : int
        Number of elements along Y-axis
    elm : Element object
        The element definitions   

     Returns
     -------
    X-coordinates of nodes, Y-Coordinates of nodes
     """

    elm.set_n_nodes((nx, ny))

    elm_width = float(c2x - c1x)
    elm_heigt = float(c2y - c1y)

    # Scale one element to element width and height
    nodes_x = elm.ctrl_pos_e * elm_width
    nodes_y = elm.ctrl_pos_n * elm_heigt

    # Generate elements
    elements = list(zip((nodes_y), (nodes_x)))

    # Unpack nodes
    ynod, xnod = zip(*elements)

    # Shift nodes to global frame
    node_x = np.array(xnod) + c1x
    node_y = np.array(ynod) + c1y

    con_matrix = np.zeros((nx * ny, 1), dtype=np.int)
    con_matrix[:, 0] = np.arange(nx * ny, dtype=np.int)

    return con_matrix, node_x, node_y


class Mesher(object):
    def __init__(self, deg_e=1, deg_n=1, type="q4"):

        """
        Mesher utility

        The Mesher is used to generate an Mesh object and provides a lightweigt GUI.



         Parameters
         ----------
        deg_e : int
            The polynomial degree in the e-direction
        deg_n : int
            The polynomial degree in the n-direction

         Returns
         -------
        Mesh :  Mehser object
         """

        self.deg_e = deg_e
        self.deg_n = deg_n
        self.type = type

    def __gui__(self):
        from matplotlib.widgets import Button, RectangleSelector
        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 8

        def render_mesh():
            try:
                data.set_xdata(
                    self._mesh_.xnodes.transpose())
                data.set_ydata(
                    self._mesh_.ynodes.transpose())
                fig.canvas.draw()
            except:
                print('Could not render mesh')
                pass

        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            self._mesh_.Xc1 = min([x1, x2])
            self._mesh_.Xc2 = max([x1, x2])
            self._mesh_.Yc1 = min([y1, y2])
            self._mesh_.Yc2 = max([y1, y2])

            self._mesh_.gen_node_positions()
            render_mesh()

        def toggle_selector(event):

            if event.key in ['W', 'w']:
                self._mesh_.n_ely += 1

            if event.key in ['X', 'x']:
                self._mesh_.n_ely -= 1

            if event.key in ['A', 'a']:
                self._mesh_.n_elx += 1

            if event.key in ['D', 'd']:
                self._mesh_.n_elx -= 1

            if event.key in ['up']:
                self._mesh_.Yc1 -= 1
                self._mesh_.Yc2 -= 1

            if event.key in ['down']:
                self._mesh_.Yc1 += 1
                self._mesh_.Yc2 += 1

            if event.key in ['left']:
                self._mesh_.Xc1 -= 1
                self._mesh_.Xc2 -= 1

            if event.key in ['right']:
                self._mesh_.Xc1 += 1
                self._mesh_.Xc2 += 1

            try:
                self._mesh_.gen_node_positions()
                render_mesh()
                pass

            except:
                pass

        def print_instructions():
            print(
                'Use arraow keys to move mesh, W and X to change refinement in  Y-directions, A and D to change refinement in X-direction')

        def ok(event):
            plt.close()

        plt.ioff()
        fig = plt.figure()

        # Doing some layout with subplots:
        fig.subplots_adjust(0.05, 0.05, 0.98, 0.98, 0.1)
        overview = plt.subplot2grid((12, 4), (0, 0), rowspan=11, colspan=4)

        n, m = self.image.shape
        overview.imshow(self.image, cmap=plt.cm.gray, origin="lower", extent=(0, m, 0, n))

        data, = overview.plot([], [], 'ro')
        overview.autoscale(1, 'both', 1)

        but_ax1 = plt.subplot2grid((12, 4), (11, 2), colspan=1)
        ok_button = Button(but_ax1, 'OK')
        ok_button.on_clicked(ok)

        but_ax2 = plt.subplot2grid((12, 4), (11, 3), colspan=1)
        reset_button = Button(but_ax2, 'Reset')

        rectangle = RectangleSelector(overview, line_select_callback,
                                      drawtype='box', useblit=True,
                                      button=[1, 3],  # don't use middle button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels')

        fig.canvas.mpl_connect('key_press_event', toggle_selector)

        _widgets = [rectangle, reset_button, ok_button]
        print_instructions()

        plt.show(block=True)

    def mesh(self, images, Xc1=0.0, Xc2=100.0, Yc1=0.0, Yc2=100., n_elx=4, n_ely=4, GUI=True, **kwargs):
        if isinstance(images, (ImageStack)):
            self.image = images[0]
        else:
            raise TypeError("Images should be in an ImageReader instance")

        if not type(Xc1) == float and type(Xc2) == float and type(Yc1) == float and type(Yc2) == float:
            raise TypeError("Coordinates should be given as floats")

        if not type(n_elx) == int and type(n_ely) == int:
            raise TypeError("Coordinates should be given as floats")

        if self.type == "spline":

            element = BSplineSurface(self.deg_e, self.deg_n, **kwargs)

        else:
            element = Q4()

        self._mesh_ = MeshStructured(element, Xc1, Xc2, Yc1, Yc2, n_elx, n_ely)

        if GUI:
            self.__gui__()

        return copy(self._mesh_)


class Mesh(object):
    def __init__(self, element, xnodes, ynodes, con_mat):
        self.element_def = element

        self.xnodes = xnodes
        self.ynodes = ynodes
        self.ele = con_mat

        self.n_nodes = len(xnodes)
        self.n_elms = np.shape(con_mat)[1]

    def scale_mesh_y(self, factor):
        """
        Scale mesh in the y direction by a factor


         Parameters
         ----------
        factor : float
            The factor which the mesh is scaled by in the y direction

         """
        center = (np.max(self.ynodes) + np.min(self.ynodes)) / 2.
        self.ynodes = factor * (self.ynodes - center) + center

    def scale_mesh_x(self, factor):
        """
        Scale mesh in the x direction by a factor


         Parameters
         ----------
        factor : float
            The factor which the mesh is scaled by in the x direction

         """
        center = (np.max(self.xnodes) + np.min(self.xnodes)) / 2.
        self.xnodes = factor * (self.xnodes - center) + center

    def center_mesh_at(self, center_point_x, center_point_y):
        """
        Center the mesh at coordinates


         Parameters
         ----------
        center_pointx : float
            The center point of the mesh in the x-direction
        center_pointy : float
            The center point of the mesh in the y-direction
         """
        center_x = (np.max(self.xnodes) + np.min(self.xnodes)) / 2.
        center_y = (np.max(self.ynodes) + np.min(self.ynodes)) / 2.

        shift_x = center_x - center_point_x
        shift_y = center_y - center_point_y

        self.xnodes = self.xnodes - shift_x
        self.ynodes = self.ynodes - shift_y


class MeshStructured(object):
    def __init__(self, element, corner1_x, corner2_x, corner1_y, corner2_y, n_elx, n_ely):
        """
        Mesh class

        Generates a grid based on the provided Finite Element definitions and geometrical measures.
        The class contains methods for generating the grid and for moving and resizing the grid.

         Parameters
         ----------
        element : object
            Instance of FiniteElement containing element definitions
        Xc1 : float
            X-Coordinate of upper left corner
        Yc1 : float
            Y-Coordinate of upper left corner
        Xc2 : float
            X-Coordinate of lower right corner
        Yc2 : float
            Y-Coordinate of lower right corner
        n_elx : int
            Number of elements in the x-direction
        n_ely : int
            Number of elements in the y-direction
         Returns
         -------
        Mesh :  Mesh object
         """
        self.element_def = element

        self.Xc1 = corner1_x
        self.Xc2 = corner2_x
        self.Yc1 = corner1_y
        self.Yc2 = corner2_y

        self.n_elx = n_elx
        self.n_ely = n_ely

        # Fields that are set after gen_mesh is called
        self.xnodes = None
        self.ynodes = None
        self.n_nodes = None
        self.n_elms = None
        self.ele = None

        self.gen_node_positions()

    def gen_node_positions(self):
        logger = logging.getLogger(__name__)
        try:
            if isinstance(self.element_def, Q4):
                logger.info("Using Q4 elements")
                self.ele, self.xnodes, self.ynodes = make_grid_Q4(self.Xc1, self.Yc1, self.Xc2, self.Yc2,
                                                                  self.n_elx,
                                                                  self.n_ely, self.element_def)

                logger.info('Element contains %.1f X %.1f pixels and is divided in %i X %i ' % (
                    (self.Xc2 - self.Xc1) / self.n_elx, (self.Yc2 - self.Yc1) / self.n_ely, self.n_elx, self.n_ely))

                self.n_nodes = len(self.xnodes)
                self.n_elms = self.n_elx * self.n_ely
            elif isinstance(self.element_def, BSplineSurface):
                logger.info("Using B-Spline elements")
                self.ele, self.xnodes, self.ynodes = make_grid(self.Xc1, self.Yc1, self.Xc2, self.Yc2,
                                                               self.n_elx,
                                                               self.n_ely, self.element_def)

                logger.info('Element contains %.1f X %.1f pixels and is divided in %i X %i ' % (
                    (self.Xc2 - self.Xc1) / self.n_elx, (self.Yc2 - self.Yc1) / self.n_ely, self.n_elx, self.n_ely))

                self.n_nodes = len(self.xnodes)
                self.n_elms = 1

            else:
                raise ValueError("Unknown element type")

        except Exception as e:
            logger.exception("Mesh generation failed")

    def scale_mesh_y(self, factor):
        """
        Scale mesh in the y direction by a factor


         Parameters
         ----------
        factor : float
            The factor which the mesh is scaled by in the y direction

         """
        center = (self.Yc2 + self.Yc1) / 2.
        height = self.Yc2 - self.Yc1
        self.Yc1 = center + (height / 2.) * factor
        self.Yc2 = center - (height / 2.) * factor

    def scale_mesh_x(self, factor):
        """
        Scale mesh in the x direction by a factor


         Parameters
         ----------
        factor : float
            The factor which the mesh is scaled by in the x direction

         """
        center = (self.Xc2 + self.Xc1) / 2.
        height = self.Xc2 - self.Xc1
        self.Xc1 = center + (height / 2.) * factor
        self.Xc2 = center - (height / 2.) * factor

    def center_mesh_at(self, center_point_x, center_point_y):
        """
        Center the mesh at coordinates


         Parameters
         ----------
        center_pointx : float
            The center point of the mesh in the x-direction
        center_pointy : float
            The center point of the mesh in the y-direction
         """
        width = self.Xc2 - self.Xc1
        height = self.Yc2 - self.Yc1
        self.Xc1 = center_point_x - (width / 2.)
        self.Xc2 = center_point_x + (width / 2.)
        self.Yc1 = center_point_y - (height / 2.)
        self.Yc2 = center_point_y + (height / 2.)

    def single_element_mesh(self):
        """
        Convert mesh to a single element mesh
        """
        self.n_elx = self.element_def.degree_e + 1
        self.n_ely = self.element_def.degree_n + 1
        self.n_elms = 1
        self.gen_node_positions()
