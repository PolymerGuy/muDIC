from unittest import TestCase

import numpy as np

from muDIC import Fields


class TestDIC_Post(TestCase):
    def test__true_strain_(self):
        # Tolerance
        toll = 1e-7
        rand_nrs = (np.random.random_sample(1000)) + 0.5

        # Format as [nEl,i,j,...]
        F = np.reshape(rand_nrs, (5, 2, 2, 5, 5, 2))
        # Calculate true strain
        U = Fields._polar_decomposition_(F)

        true_strain = Fields._true_strain_(U)
        # Determine absolute error
        self.fail()

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

        fields = Fields(F,None)
        # Green strain to be tested
        G = fields.green_strain()
        print(F.shape)
        print(G.shape)


        # Determine absolute error
        deviation = np.abs(Green_strain - G)
        print(deviation)

        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_simple_shear(self):
        """
       Simple shear test

       Tested for true strain, engineering strain and green strain

        NOTE THAT THIS ONLY FOLDS FOR VERY SMALL SHEAR STRAINS

       It is checked that:
       eng_strain_11 = U_11 -1
       true_strain_11 = log(U_11)
       green_strain_11 = 0.5(U_11^2 -1)
       """
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]

        a = 0.0002

        # Format as [nEl,i,j,...]
        F = np.array([[1,a],[0,1]])
        # Calculate green deformation as F^T*F

        F = F[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]


        fields = Fields(F,None)



        eng_strain = fields.eng_strain()
        print(eng_strain[0,:,:,0,0,-1])
        eng_strain_correct = np.zeros_like(eng_strain)
        eng_strain_correct[:, 0, 1, :, :, :] = 0.5*a
        eng_strain_correct[:, 1, 0, :, :, :] = 0.5*a
        deviation = np.abs(eng_strain_correct - eng_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

        true_strain = fields.true_strain()
        print(np.exp(true_strain.max()))
        true_strain_correct = np.zeros_like(true_strain)
        true_strain_correct[:, 0, 1, :, :, :] = 0.5 * np.log(1. + a)
        true_strain_correct[:, 1, 0, :, :, :] = 0.5 * np.log(1. + a)
        deviation = np.abs(true_strain_correct - true_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))




        green_strain = fields.green_strain()
        print(green_strain.max())

        # By hand calculation this should give
        green_strain_correct = 0.5 * np.array([[a**2,a],[a,0]])[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        deviation = np.abs(green_strain_correct - green_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_pure_shear(self):
        """
       Pure shear test

       Tested for true strain, engineering strain and green strain

       It is checked that:
       eng_strain_11 = U_11 -1
       true_strain_11 = log(U_11)
       green_strain_11 = 0.5(U_11^2 -1)
       """
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]

        a = 0.2

        # Format as [nEl,i,j,...]
        F = np.array([[1, a], [a, 1]])
        # Calculate green deformation as F^T*F

        F = F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        fields = Fields(F, None)

        eng_strain = fields.eng_strain()
        eng_strain_correct = np.zeros_like(eng_strain)
        eng_strain_correct[:, 0, 1, :, :, :] = a
        eng_strain_correct[:, 1, 0, :, :, :] = a
        deviation = np.abs(eng_strain_correct - eng_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

        true_strain = fields.true_strain()
        print(np.exp(true_strain.max()))
        true_strain_correct = np.zeros_like(true_strain)
        true_strain_correct[:, 0, 1, :, :, :] = np.log(1. + a)
        true_strain_correct[:, 1, 0, :, :, :] = np.log(1. + a)
        deviation = np.abs(true_strain_correct - true_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

        green_strain = fields.green_strain()
        print(green_strain.max())

        # By hand calculation this should give
        green_strain_correct = 0.5 * np.array([[a ** 2, a], [a, 0]])[np.newaxis, :, :, np.newaxis, np.newaxis,
                                     np.newaxis]
        deviation = np.abs(green_strain_correct - green_strain)
        self.assertEqual(True, all([dev < toll for dev in deviation.flatten()]))

    def test_engineering_strain_(self):
        # Tolerance
        toll = 1e-7
        # Generate random numbers in [0.5,1.5]
        rand_nrs = (np.random.random_sample(2500)) + 0.5
        # Format as [nEl,i,j,...]
        theta = np.pi/5 * 0.
        R = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        U = np.array([[1.5,0.0000],[0.,-0.5]])
        F = R.dot(U)
        F = F[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]

        fields = Fields(F,None)
        green_strain = fields.green_strain()

        eng_strain = fields.eng_strain()

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
        """
         Stretch along the x-axis

         This holds for true strain, engineering strain and green strain

         It is checked that:
         eng_strain_11 = U_11 -1
         true_strain_11 = log(U_11)
         green_strain_11 = 0.5(U_11^2 -1)
         """

        # Tolerance
        toll = 1e-7

        # Deformation gradient
        stretch = 1.1
        F = np.array([[stretch, 0.], [0., 1.]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        fields = Fields(F_stack,None)

        eng_strain = fields.eng_strain()
        eng_strain_correct = np.zeros_like(eng_strain)
        eng_strain_correct[:, 0, 0, :, :, :] = stretch - 1.0
        deviation_eng_strain = np.abs(eng_strain - eng_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_eng_strain.flatten()]))

        true_strain = fields.true_strain()
        true_strain_correct = np.zeros_like(true_strain)
        true_strain_correct[:, 0, 0, :, :, :] = np.log(stretch)
        deviation_true_strain = np.abs(true_strain - true_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_true_strain.flatten()]))

        green_strain = fields.green_strain()
        green_strain_correct = np.zeros_like(true_strain)
        green_strain_correct[:, 0, 0, :, :, :] = 0.5 * (stretch ** 2. - 1.)
        deviation_green_strain = np.abs(green_strain - green_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_green_strain.flatten()]))


    def test_rotation_(self):
        """
        Pure rotation should not induce spurious strains
        This holds for true strain, engineering strain and green strain
        """
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        theta = np.pi/6.
        F = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        fields = Fields(F_stack,None)

        # Should only contain zeros
        eng_strain = fields.eng_strain()
        self.assertEqual(True, all([dev < toll for dev in eng_strain.flatten()]))

        true_strain = fields.true_strain()
        self.assertEqual(True, all([dev < toll for dev in true_strain.flatten()]))

        green_strain = fields.green_strain()
        self.assertEqual(True, all([dev < toll for dev in green_strain.flatten()]))





    def test_tension_rotation_(self):
        """
        Stretch along the x-axis, rotate so that its aligned along y.
        Should still only give a strain component along x.

        This holds for true strain, engineering strain and green strain

        The deformation gradient is assembled as F=RU where U_11 is defined and R corresponds to a 90deg rotation.

        It is checked that:
        eng_strain_11 = U_11 -1
        true_strain_11 = log(U_11)
        green_strain_11 = 0.5(U_11^2 -1)
        """
        # Tolerance
        toll = 1e-7

        # The deformation gradient is calculated as F=RU
        stretch = 1.5
        theta = np.pi/2.
        R = np.array([[np.cos(theta),np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        U = np.array([[stretch,0],[0,1.]])
        F = R.dot(U)
        # We need to have the right formatting
        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        fields = Fields(F_stack,None)

        eng_strain = fields.eng_strain()
        eng_strain_correct = np.zeros_like(eng_strain)
        eng_strain_correct[:,0,0,:,:,:] = stretch-1.0
        deviation_eng_strain = np.abs(eng_strain-eng_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_eng_strain.flatten()]))


        true_strain = fields.true_strain()
        true_strain_correct = np.zeros_like(true_strain)
        true_strain_correct[:,0,0,:,:,:] = np.log(stretch)
        deviation_true_strain = np.abs(true_strain-true_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_true_strain.flatten()]))


        green_strain = fields.green_strain()
        green_strain_correct = np.zeros_like(true_strain)
        green_strain_correct[:,0,0,:,:,:] = 0.5 * (stretch**2.-1.)
        deviation_green_strain = np.abs(green_strain-green_strain_correct)
        self.assertEqual(True, all([dev < toll for dev in deviation_green_strain.flatten()]))





    def test_biaxial_tension_(self):
        # Tolerance
        toll = 1e-7

        # Deformation gradient
        F = np.array([[1.1, 0.], [0., 1.1]])

        F_stack = np.ones((5, 2, 2, 5, 5, 2)) * F[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

        fields = Fields(F_stack,None)

        eng_strain = fields.eng_strain()

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

        fields = Fields(F_shear_stack,None)

        E_shear = fields.green_strain()

        eng_strain_shear = fields.eng_strain()

        # Rotate to tension-compression orientation
        alpha = np.pi / 4.
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        F_tc = np.dot(R.transpose(), np.dot(F_shear, R))

        F_tc_stack = np.ones((5, 2, 2, 5, 5, 2)) * F_tc[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
        fields_tc = Fields(F_tc_stack,None)

        E_tc = fields_tc.green_strain()

        eng_strain_tc = fields_tc.eng_strain()

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
