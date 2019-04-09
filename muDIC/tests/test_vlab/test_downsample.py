from unittest import TestCase

import numpy as np

from muDIC.vlab import Downsampler
from muDIC.vlab.downsampler import chess_board


class TestDownsample(TestCase):

    def test_passthrough(self):
        tol = 1e-6
        image = chess_board(10)

        downsampler = Downsampler(image_shape=image.shape, factor=1, fill=1.0, pixel_offset_stddev=0.)
        passthrough = downsampler(image)

        deviation = np.abs(image - passthrough)

        if np.max(deviation) > tol:
            self.fail("Unit scaling should return the same image. Got %f and not %f" % (passthrough[0, 0], image[0, 0]))

    def test_tile_average(self):
        """"
        The average value of the tiles should be pm 20/16=1.25
        """
        tol = 1e-6
        image = chess_board(10)

        downsampler = Downsampler(image_shape=image.shape, factor=4, fill=1.0, pixel_offset_stddev=0.)

        tile_averages = downsampler(image)
        correct_tile = np.zeros((2, 2))
        correct_tile[0, 0] = 1.25
        correct_tile[0, 1] = -1.25
        correct_tile[1, 0] = -1.25
        correct_tile[1, 1] = 1.25

        correct_tiles = np.tile(correct_tile, (10, 10))

        deviation = np.abs(tile_averages - correct_tiles)

        if np.max(deviation) > tol:
            self.fail("The average value of the tiles is not correct. Should be %f but got %f" % (
            correct_tiles[0, 1], tile_averages[0, 1]))

    def test_tile_center_average(self):
        """"
        The average value of the tiles should be pm 2.
        """
        tol = 1e-6
        image = chess_board(10)

        downsampler = Downsampler(image_shape=image.shape, factor=4, fill=.25, pixel_offset_stddev=0.)

        tile_averages = downsampler(image)
        correct_tile = np.zeros((2, 2))
        correct_tile[0, 0] = 2.
        correct_tile[0, 1] = -2.
        correct_tile[1, 0] = -2.
        correct_tile[1, 1] = 2.

        correct_tiles = np.tile(correct_tile, (10, 10))

        deviation = np.abs(tile_averages - correct_tiles)

        if np.max(deviation) > tol:
            self.fail("The average value of the center of the tiles is not correct. Should be %f but got %f" % (
            correct_tiles[0, 1], tile_averages[0, 1]))
