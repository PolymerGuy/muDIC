import numpy as np

from . import image_deformer as cordmap
from .downsampler import Downsampler


class SyntheticImageGenerator(object):
    def __init__(self, speckle_image, image_deformer, downsampler, noise_injector, n=10):
        """Virtual experiment

        This class sets up a simple pipeline for generating a set of synthetically generated
        and deformed speckle images.

        The pipeline is as follows:
        A speckle image is deformed by the image_deformer
        The deformed image is then downsampled
        The downsampled image is then added noise

        For a larger deformation, the image_deformer deforms the image more and the same pipeline is used.

        Example
        -------
        Let us generate a synthetic image generator which produced 500x500 pixel images with the following approach

        * Generate a synthetic speckle using the "Rosta" algorithm
        * Deform the base image by a given deformation gradient
        * Down-sample the image by a factor of 4, with a pixel fill factor of 0.8 and a pixel offset standard deviation
            of 0.1 pixel
        * Add gaussian additive noise to the image

        The neccessary code to do this is as follows:

            >>> n = 10
            >>> image_shape = (2000, 2000)
            >>> # speckle_image = vlab.speckle.make_perlin_noise_speckle(image_shape,octaves=3)
            >>> speckle_image = vlab.rosta_speckle(image_shape, dot_size=4, density=0.32, smoothness=2.0, layers=4)
            >>>
            >>> F = np.array([[1.1, .0], [0., 1.0]], dtype=np.float64)
            >>> image_deformer = vlab.imageDeformer_from_defGrad(F)
            >>>
            >>> downsampler = vlab.Downsampler(image_shape=image_shape, factor=4, fill=0.8, pixel_offset_stddev=0.1)
            >>>
            >>> noise_injector = vlab.noise_injector("gaussian", sigma=.01)
            >>>
            >>> image_generator = vlab.SyntheticImageGenerator(speckle_image=speckle_image, image_deformer=image_deformer,
            >>>                                          downsampler=downsampler, noise_injector=noise_injector, n=n)
            >>> image_stack = dic.ImageStack(image_generator)



        Note
        ----
        All arguments have to be provided and no default behavior is defined

        Parameters
        ----------
        Speckle_image : ndarray
            The speckle image.
        image_deformer : instance of ImageDeformer
            ImageDeformer deforms an image according to the coordinate mapper given to it upon instantiation
        downsampler: instance of Downsampler
            Downsampler downsamples an image according to the settings given to it upon instantiation
        noise_injector: noise injection function
            noise_injector is a function which takes an image and returns an image with noise
        n : int
            Number of frames to be deformed
        """

        self.n_images = n

        if isinstance(speckle_image, np.ndarray):
            if speckle_image.ndim != 2:
                raise ValueError("Only 2D arrays are accepted")
            self.speckle_image = speckle_image

        if isinstance(image_deformer, cordmap.ImageDeformer):
            self._image_deformer_ = image_deformer
        else:
            raise ValueError("Only instances of CoordinateMapper are accepted")

        # If a speckle image is provided, generate a matching down sampler
        if isinstance(downsampler, Downsampler):
            self.down_sampler = downsampler
        else:
            raise ValueError("Only instances of Downsampler are accepted")

        if noise_injector is None:
            raise ValueError("Noise_injector must be specified")



        self.noise_injector = noise_injector

        self._deformed_images_ = self._image_deformer_(self.speckle_image, steps=self.n_images)

        self._deformed_noisy_images_ = [self.noise_injector(img) for img in self._deformed_images_]

        self.phantoms = [self.down_sampler(img) for img in self._deformed_noisy_images_]

    def deformation_field(self):
        # return self._generate_F_field_(self.F, self._weight_field_)
        # return np.linalg.inv(self._F_field_)
        return self._image_deformer_.coodinate_mapper.def_grad

    def __len__(self):
        return len(self.phantoms)

    def __call__(self, img_ind):
        # TODO: Make this thing lazy
        return self.phantoms[img_ind]
