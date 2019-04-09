import logging
from unittest import TestCase

import numpy as np

import muDIC.vlab as vlab


class TestVirtualTensileTest(TestCase):
    # TODO: Rewrite these tests!
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        cls.logger = logging.getLogger()
        np.set_printoptions(precision=8)

        cls.img_shape = (500, 500)
        cls.tol = 1e-5

        cls.image = vlab.speckle.dots_speckle(cls.img_shape, n_dots=10000, dot_radius_max=10)

    def test__pass_through_user_img(self):
        F = np.eye(2, dtype=np.float)
        image_deformer = vlab.imageDeformer_from_defGrad(F)

        downsampler = vlab.Downsampler(image_shape=self.img_shape, factor=1, fill=1., pixel_offset_stddev=0.0)

        noise_injector = lambda img: img

        virtualTest = vlab.SyntheticImageGenerator(speckle_image=self.image, image_deformer=image_deformer,
                                                   downsampler=downsampler,
                                                   noise_injector=noise_injector, n=10)

        deviation = np.abs(virtualTest(1) - self.image)

        if np.max(deviation) > self.tol:
            self.fail("Image changed value or orientation. Largest error is%f" % np.max(deviation))
