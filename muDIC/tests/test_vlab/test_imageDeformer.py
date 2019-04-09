from unittest import TestCase

import numpy as np

from muDIC import vlab


class TestImageDeformer(TestCase):

    def test_defgrad_dilate_square_by_biaxial_tension(self):
        """
        Deforms a image with a square by a given amount and checks the area of the square
        """
        tol = 1e-4
        square = np.zeros((200, 200), dtype=np.float)
        square[75:125, 75:125] = 1.

        undeformed_hole_area = np.sum(square)

        F = np.array([[2., .0], [0., 2.0]], dtype=np.float64)

        image_deformer = vlab.imageDeformer_from_defGrad(F)

        enlarged_hole = image_deformer(square, steps=2)[1]
        enlarged_hole_area = np.sum(enlarged_hole)
        area_increase = enlarged_hole_area / undeformed_hole_area
        if np.abs(area_increase - np.linalg.det(F)) > tol:
            self.fail("Deformed area should be %f but was %f" % (np.linalg.det(F), area_increase))

    def test_defgrad_rotate_square(self):
        """
        Subtract an rotated and un-rotated square.
        Two cases:

        * Rotate 45deg and subtract, should give a non-zero diffence
        * Rotate 90 deg and subtract, should give a zero difference
        """
        tol = 1e-4
        square = np.zeros((200, 200), dtype=np.float)
        square[75:125, 75:125] = 1.

        theta = np.pi / 4.
        F_rot_45 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        image_deformer_45 = vlab.imageDeformer_from_defGrad(F_rot_45)
        image_rotated_45 = image_deformer_45(square, steps=2)[1]

        if np.abs(np.sum(np.abs(square - image_rotated_45))) < 1.:
            self.fail("The rotated image is the same as the un-rotated one")

        theta = np.pi / 2.
        F_rot_90 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        image_deformer_90 = vlab.imageDeformer_from_defGrad(F_rot_90)
        image_rotated_90 = image_deformer_90(square, steps=2)[1]

        if np.abs(np.sum(np.abs(square - image_rotated_90))) > tol:
            self.fail("The image rotated by 90deg should be identical to the un-rotated image")
