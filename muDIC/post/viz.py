import logging

import matplotlib.pyplot as plt
import numpy as np

from .fields import Fields


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
            plt.tricontourf(xs, ys, fvar, **kwargs)
            plt.colorbar()

        plt.title("%s component %i,%i at frame %i" % (keyword,component[0],component[1], frame))
        plt.show()


