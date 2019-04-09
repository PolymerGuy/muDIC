import logging
from unittest import TestCase

import numpy as np

import muDIC as dic
import muDIC.solver.correlate
from muDIC import vlab
from muDIC.vlab import SyntheticImageGenerator
from muDIC.vlab.downsampler import Downsampler


class TestFullDICCore(TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        np.set_printoptions(precision=8)
        cls.img_shape = (500, 500)
        cls.image = muDIC.vlab.speckle.harmonic_speckle(cls.img_shape, n_peaks_x=10)

    def __run_DIC_defgrad__(self, F, element, x1=150, x2=350, y1=150, y2=350, n_elx=3, n_ely=3):
        """
        Function that imposes a deformation gradient upon a reference image, runs a full DIC job and compares
        the resulting deformation gradient to the imposed gradient. This is done for all points within the elements.

        :param F: Deformation gradient
        :param element: Finite element definitions
        :return True if test is passed
        """

        # Default settings
        n_frames = 10
        rel_tol_F = 1e-2

        downsampler = Downsampler(image_shape=self.img_shape, factor=1, fill=1.0, pixel_offset_stddev=0.0)

        image_deformer = vlab.imageDeformer_from_defGrad(F)

        noise_injector = lambda img: img

        image_reader = SyntheticImageGenerator(speckle_image=self.image, image_deformer=image_deformer,
                                               downsampler=downsampler, noise_injector=noise_injector, n=n_frames)
        images_stack = dic.IO.ImageStack(image_reader)

        # Initialize a mesh instance
        if isinstance(element, dic.elements.BSplineSurface):
            mesher_tool = dic.mesh.Mesher(element.degree_e, element.degree_n)

            myMesh = mesher_tool.mesh(images_stack, x1, x2, y1, y2, n_elx, n_ely, GUI=False)
        else:
            raise TypeError("Invalid element received")

        # Generate mesh

        RefUpdate = [5]

        settings = muDIC.solver.correlate.DICInput(myMesh, images_stack, RefUpdate)
        settings.tol = 1e-6
        # Run DIC

        analysis = dic.solver.DICAnalysis(settings)

        dic_results = analysis.run()

        n_nodes, n_frames_dic = np.shape(dic_results.xnodesT)

        if n_frames != n_frames_dic:
            return False

        results = dic.Fields(dic_results, seed=21)

        max_error, mean_error = self.__calculate_DIC_error__(F, results.F())
        print("stuff:", max_error, mean_error)

        return max_error <= rel_tol_F

    def __calculate_DIC_error__(self, F_correct, F_dic):
        # TODO: Possible bug in the above line?
        # TODO: 12 is here 21. Skipping first frame due to gopher in the post process

        approx_zero = 1.e-5
        F11 = F_dic[:, 0, 0, :, :, :]
        F12 = F_dic[:, 0, 1, :, :, :]
        F21 = F_dic[:, 1, 0, :, :, :]
        F22 = F_dic[:, 1, 1, :, :, :]

        n_frames = F_dic.shape[-1]

        F_correct = F_correct

        # Make empty matrix for storing deformation gradients
        F_correct_at_frame = np.zeros((n_frames, 2, 2))
        F_correct_at_frame[0, :, :] = np.eye(2, dtype=np.float)

        print("F Correct:", F_correct)

        for i in range(n_frames - 1):
            F_correct_at_frame[i + 1] = np.dot(F_correct_at_frame[i], F_correct)

        F_correct_deviatior = F_correct_at_frame - np.eye(2, dtype=np.float)[np.newaxis, :, :]

        F11_abs_error = np.zeros((F11.shape[1], F11.shape[2], F11.shape[3]))
        F12_abs_error = np.zeros_like(F11_abs_error)
        F21_abs_error = np.zeros_like(F11_abs_error)
        F22_abs_error = np.zeros_like(F11_abs_error)

        # Determine error between imposed deformation gradient and DIC results
        # If the correct component is 0, only the absolute difference is calculated
        # If the correct component is not 0, the absolute relative error i calculated
        for t in range(n_frames):

            if np.abs(F_correct_deviatior[t, 0, 0]) > approx_zero:
                print("Reporting relative error for F11")
                F11_abs_error[:, :, t] = np.abs(
                    (F11[0, :, :, t] - F_correct_at_frame[t, 0, 0]) / F_correct_deviatior[t, 0, 0])
            else:
                F11_abs_error[:, :, t] = np.abs((F11[0, :, :, t] - F_correct_at_frame[t, 0, 0]))

            if np.abs(F_correct_deviatior[t, 0, 1]) > approx_zero:
                print("Reporting relative error for F12, being: ", F_correct_deviatior[t, 0, 1])

                F12_abs_error[:, :, t] = np.abs(
                    (F12[0, :, :, t] - F_correct_at_frame[t, 0, 1]) / F_correct_deviatior[t, 0, 1])
            else:
                F12_abs_error[:, :, t] = np.abs(F12[0, :, :, t] - F_correct_at_frame[t, 0, 1])

            if np.abs(F_correct_deviatior[t, 1, 0]) > approx_zero:
                print("Reporting relative error for F21")

                F21_abs_error[:, :, t] = np.abs(
                    (F21[0, :, :, t] - F_correct_at_frame[t, 1, 0]) / F_correct_deviatior[t, 1, 0])
            else:
                F21_abs_error[:, :, t] = np.abs(F21[0, :, :, t] - F_correct_at_frame[t, 1, 0])

            if np.abs(F_correct_deviatior[t, 1, 1]) > approx_zero:
                print("Reporting relative error for F22")

                F22_abs_error[:, :, t] = np.abs(
                    (F22[0, :, :, t] - F_correct_at_frame[t, 1, 1]) / F_correct_deviatior[t, 1, 1])
            else:
                F22_abs_error[:, :, t] = np.abs(F22[0, :, :, t] - F_correct_at_frame[t, 1, 1])

        print('Largest error in gradient F11:', np.max(F11_abs_error[:, :, :]))
        # print 'Largest error in gradient F11 for each frame:', np.max(F11_abs_error[:, :, :],axis=(0,1))
        print('Largest error in gradient F12:', np.max(F12_abs_error[:, :, :]))
        # print 'Largest error in gradient F12 for each frame:', np.max(F12_abs_error[:, :, :],axis=(0,1))

        print('Largest error in gradient F21:', np.max(F21_abs_error[:, :, :]))
        print('Largest error in gradient F22:', np.max(F22_abs_error[:, :, :]))
        # print 'Largest error in gradient F11 field:', np.argmax(F11_abs_error[:, :, :],axis=2)
        # plt.figure()
        # plt.imshow(F11[0,:,:,1])
        print("F11_correct is:%f" % F_correct_at_frame[1, 0, 0])

        # plt.show()

        max_abs_error = np.max(
            [np.max(F11_abs_error), np.max(F12_abs_error), np.max(F21_abs_error), np.max(F22_abs_error)])

        mean_abs_error = None
        return max_abs_error, mean_abs_error

    def test_deg1_rotation(self):
        element = dic.elements.BSplineSurface(deg_e=1, deg_n=1)

        theta = 0.01
        F = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        # print F

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg1_rotation_biaxial(self):
        element = dic.elements.BSplineSurface(deg_e=1, deg_n=1)

        theta = 0.01
        F_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        F_biax = np.array([[1.001, 0.], [0., 0.999]], dtype=np.float64)
        F = np.dot(F_rot, F_biax)
        # print F

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg1_biaxial(self):
        element = dic.elements.BSplineSurface(deg_e=1, deg_n=1)
        F = np.array([[1.001, 0.], [0., 0.999]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg1_simple_shear(self):
        element = dic.elements.BSplineSurface(deg_e=1, deg_n=1)
        F = np.array([[1.00, .005], [0., 1.]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg2_rotation(self):
        element = dic.elements.BSplineSurface(deg_e=2, deg_n=2)

        theta = 0.01
        F = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        # print F

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg2_biaxial(self):
        element = dic.elements.BSplineSurface(deg_e=2, deg_n=2)
        F = np.array([[1.001, 0.], [0., .999]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg2_simple_shear(self):
        element = dic.elements.BSplineSurface(deg_e=2, deg_n=2)
        F = np.array([[1.00, .005], [0., 1.]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg3_rotation(self):
        element = dic.elements.BSplineSurface(deg_e=3, deg_n=3)

        theta = 0.01
        F = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        # print F

        passed = self.__run_DIC_defgrad__(F, element, x1=100, x2=300, y1=100, y2=300, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg3_biaxial(self):
        element = dic.elements.BSplineSurface(deg_e=3, deg_n=3)
        F = np.array([[1.001, 0.], [0., .999]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)

    def test_deg3_simple_shear(self):
        element = dic.elements.BSplineSurface(deg_e=3, deg_n=3)
        F = np.array([[1.00, .005], [0., 1.]], dtype=np.float64)

        passed = self.__run_DIC_defgrad__(F, element, n_elx=4, n_ely=4)
        self.assertEqual(passed, True)
