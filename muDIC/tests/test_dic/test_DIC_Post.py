from unittest import TestCase

import numpy as np

from muDIC import Fields


class TestDIC_Post(TestCase):
    def test__true_strain_(self):
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [-0.99,4.]
        rand_nrs = 5. * (np.random.random_sample(1000)) - 0.99
        # Format as [nEl,i,j,...]
        eng_strain = np.reshape(rand_nrs, (5, 2, 2, -1))
        # Calculate true strain
        true_strain = Fields._true_strain_(eng_strain)
        # Determine absolute error
        deviation = np.abs(np.log(eng_strain + 1.) - true_strain)
        # Pass if all elements are within tolerance
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_green_strain_(self):
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]
        rand_nrs = (np.random.random_sample(1000)) + 0.5

        # Format as [nEl,i,j,...]
        F = np.reshape(rand_nrs, (5, 2, 2, 5, 5, 2))
        # Calculate green deformation as F^T*F
        Green_deformation = np.einsum('nji...,njo...->nio...', F, F)

        I = np.eye(2, dtype=np.float)

        # Calculate green strain tensor as 0.5(F^T * F - I)
        Green_strain = 0.5 * (Green_deformation - I[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis])

        # Green strain to be tested
        G = Fields._green_strain_(F)
        print(F.shape)
        print(G.shape)


        # Determine absolute error
        deviation = np.abs(Green_strain - G)
        print(deviation)

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_green_strain_shear(self):
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]

        a = 0.2

        # Format as [nEl,i,j,...]
        F = np.array([[1,0],[a,1]])
        # By hand calculation this should give
        green_strain = 0.5 * np.array([[a**2,a],[a,0]])[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        # Calculate green deformation as F^T*F

        F = F[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]

        # Green strain to be tested
        G = Fields._green_strain_(F)

        # Determine absolute error
        deviation = np.abs(green_strain - G)
        print("Deviation is:",deviation)
        print(green_strain[0,:,:,0,0,0])
        print(G[0,:,:,0,0,0])

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))



    def test_engineering_strain_(self):
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]
        rand_nrs = (np.random.random_sample(2500)) + 0.5
        # Format as [nEl,i,j,...]
        F = np.reshape(rand_nrs, (5, 2, 2, 5, 5, 5))

        theta = np.pi/5 * 0.
        R = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        U = np.array([[1.5,0.0000],[0.,-0.5]])
        F = R.dot(U)
        F = F[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]

        green_strain = Fields._green_strain_(F)

        eng_strain = Fields._engineering_strain_(green_strain)

        # Calculate green strain from engineering strain
        E11 = 0.5 * ((eng_strain[:, 0, 0, :, :, :] + 1.) ** 2. - 1.)
        E22 = 0.5 * ((eng_strain[:, 1, 1, :, :, :] + 1.) ** 2. - 1.)
        E12 = 0.5 * np.sin(2. * eng_strain[:, 0, 1, :, :, :]) * (1. + eng_strain[:, 0, 0, :, :, :]) * (
                1. + eng_strain[:, 1, 1, :, :, :])



        # Deterine absolute error
        deviation11 = np.abs(E11 - green_strain[:, 0, 0, :, :, :])
        deviation22 = np.abs(E22 - green_strain[:, 1, 1, :, :, :])
        deviation12 = np.abs(E12 - green_strain[:, 0, 1, :, :, :])


        print(deviation11)

        self.assertEqual(True, all([dev < toll for dev in deviation11.flatten()]))
        self.assertEqual(True, all([dev < toll for dev in deviation12.flatten()]))
        self.assertEqual(True, all([dev < toll for dev in deviation22.flatten()]))

    def test_uniaxial_tension_(self):
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        F = np.array([[1.1, 0.], [0., 1.]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E = Fields._green_strain_(F_stack)

        eng_strain = Fields._engineering_strain_(E)

        eng_strain - (F - np.eye(2))[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        # Determine absolute error
        deviation = np.abs(eng_strain - (F - np.eye(2))[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis])

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_pure_rotation_(self):
        """
        Pure rotation should not induce spurious strains!
        :return:
        """
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        theta = np.pi/6.
        F = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E = Fields._green_strain_(F_stack)

        eng_strain = Fields._engineering_strain_(E)

        deviation = eng_strain

        # Determine absolute error
        #self.assertEqual(eng_strain,np.zeros_like(eng_strain))

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))



    def test_pure_tension_rotation_(self):
        """
        Stretch along x, rotate so that its aligned along y
        :return:
        """
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        theta = np.pi/2.
        R = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        U = np.array([[1.5,0],[0,1.]])
        F = R.dot(U)
        print(F)

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E = Fields._green_strain_(F_stack)

        eng_strain = Fields._engineering_strain_(E)
        eng_strain_right = np.zeros_like(eng_strain)
        eng_strain_right[:,0,0,:,:,:] = 0.5

        deviation = np.abs(eng_strain-eng_strain_right)

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))




    def test_biaxial_tension_(self):
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        F = np.array([[1.1, 0.], [0., 1.1]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E = Fields._green_strain_(F_stack)

        eng_strain = Fields._engineering_strain_(E)

        eng_strain - (F - np.eye(2))[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        # Determine absolute error
        deviation = np.abs(eng_strain - (F - np.eye(2))[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis])

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_shear_(self):
        # Small strain pure shear compared to rotated tension-compression

        # Tolerance
        toll = 1e-7

        # Deformation gradient
        F_shear = np.array([[1., 0.001], [0.001, 1.]])

        F_shear_stack = np.ones((5, 2, 2, 5, 5, 2)) * F_shear[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E_shear = Fields._green_strain_(F_shear_stack)

        eng_strain_shear = Fields._engineering_strain_(E_shear)

        # Rotate to tension-compression orientation
        alpha = np.pi / 4.
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        F_tc = np.dot(R.transpose(), np.dot(F_shear, R))

        F_tc_stack = np.ones((5, 2, 2, 5, 5, 2)) * F_tc[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        E_tc = Fields._green_strain_(F_tc_stack)

        eng_strain_tc = Fields._engineering_strain_(E_tc)

        # Rotate back to pure shear
        alpha = -np.pi / 4.
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        eng_strain_shear_rot = np.einsum('ij,njk...,kl...->nil...', R.transpose(), eng_strain_tc, R)

        deviation = np.abs(eng_strain_shear - eng_strain_shear_rot)

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

# TODO: Write this test!
#    def rotated_tension_(self):
#        # Tolerance
#        toll = 1e-7
#
#        # Deformation gradient
#        F_tension = np.array([[1.1,0.],[0.,1.]])
#
#        # Rotate back to pure shear
#        alpha = -np.pi/4.
#        R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
#
#        F_tension_rotation = np.dot(R,F_tension)
#
#        F_stack = np.ones((5,2,2,5,5,2)) * F_tension_rotation[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
#
#        E = DIC_Post._green_strain_(F_stack)
#
#        eng_strain = DIC_Post._engineering_strain_(E)
#
#        alpha = np.pi/4.
#        R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
#        eng_strain_shear_rot = np.einsum('ij,njk...,kl...->nil...', R.transpose(), eng_strain_tc, R)
#
#        # Determine absolute error
#        deviation = np.abs(eng_strain - (F-np.eye(2))[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis])
#
#        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))
