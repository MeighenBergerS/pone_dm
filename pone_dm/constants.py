# -*- coding: utf-8 -*-
# Name: constants.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Contains all required constants

# Imports
import numpy as np


class pdm_constants(object):
    """ stores all relevant constants for the package
    """

    def __init__(self):
        # ----------------------------------------------------------------------
        # '# ##' == used in the simulation code----
        self.h = 0.71
        self.H_0 = 100 * self.h  # h km s^-1 Mpc^-1 hubble time --- # ##
        self.H_0 = self.H_0 * 1e5 / 3.086e24  # h s^-1 ## Mpc->cm km-»cm
        # self.rho_c = 2.7754e11 * self.h**2  # M_0 Mpc^-3------> Mpc!!!!!
        # self.rho_c = self.rho_c * (1.9e30 * 1.78e-27**(-1)) * (3.086e24)**(-3)
        self.rho_c = 1.053e-5 * self.h**2  # GeV cm^-3 -----------------------!!!!
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
          

        #self.G_N = 6.67e-11  # m^3 kg^-1 s^-1
        #self.G_N = self.G_N * 1e6 * 1e-22 / 1.78  # cm^3 (GeV/c^2)^-1 s^-1 ## m->cm kg->GeV/c^2
        #self.rho_c = (3 * self.H_0**2) / (8 * np.pi * self.G_N)

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
