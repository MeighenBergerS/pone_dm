# -*- coding: utf-8 -*-
# Name: constants.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Contains all required constants

# Imports

class pdm_constants(object):
    """ stores all relevant constants for the package
    """

    def __init__(self):
        # ----------------------------------------------------------------------
        # '# ##' == used in the simulation code----
        # self.rho_c_mpc = 2.7754e11 * 0.7**2 # h^-2 M_0 Mpc^-3------> Mpc!!!!!
        self.rho_c_mpc = 5.5e-6  # GeV cm^-3 ----------------------- !!!!!!!
        self.gamma = 1.3186  # NFW parameter --- slope parameter -----  # ##
        self._kappa = 2
        # ----------------------------------------------------------------------
        # P-ONE
        # values of  J for P-ONE ------------
        # c_zenith_1= [-1,-0.5]
        self.J_s1 = 0.87e23  # ##
        self.J_p1 = 0.85e17     # ##
        self.J_d1 = 1.4e11   # ##
        # c_zenith_2= [-0.5,0   # ##.5]
        self.J_s2 = 1.2e23   # ##
        self.J_p2 = 1.2e17   # ##
        self.J_d2 = 2.0e11   # ##
        # c_zenith_3= [0.5,1]   # ##
        self.J_s3 = 0.13e23   # ##
        self.J_p3 = 0.13e17   # ##
        self.J_d3 = 0.18e11   # ##

        # All sky J - Values [ Crlos et.al - DM Annihiliation to Neutrinos]
        self.J_s = 2.3e23  # ##
        self.J_p = 2.2e17  # ##
        self.J_d = 3.6e11  # ##
        # ----------------------------------------------------------------------
        self.h = 0.71
        self.H_0 = 100 * self.h  # h km s^-1 Mpc^-1 hubble time --- # ##

        self._omega_m = 0.27   # ##
        self._omega_L = 0.721  # ##
        self.omega_r = 4.75e-5  # ##
        self.omega_DM = 0.23   # ##
        # self.omega_c = 0.2589

        # self.omega_B = 0.045
        # self.omega_k = -0.09

        # ----------------------------------------------------------------------
        self.Delta = 200  # Lopez et al. ---------- M_min=10e-5------  # ##
        self.msq2cmsq = 1e4  # Converts m^2 to cm^2 # ##

        # for particular profiles I used these which are in kpc and
        # their units cancels out so not much of difference but still should
        # be checked again

        self.r_s = 20  # kpc,

        self.rho_0 = 0.4  # GeV/cm**3

        self.R_0 = 0.8134  # kpc

        # The concentation parameters for DM halos
        self.c_200 = 100  # Parada et. al
        self.delta_c = 1.686  # Diemer et. al 2015

    @property
    def omega_m(self):
        return self._omega_m

    @property
    def omega_L(self):
        return self._omega_L
