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
        self.rho_c_mpc = 2.7754e11  # h^-2 M_0 Mpc^-3-------------> Mpc!!!!!  # ##
        self.gamma = 1.3186  # NFW parameter --- slope parameter -----  # ##
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
        self.H_0 = 71  # km/(Mpc*s) hubble time --- # ##
        self.omega_k = -0.09
        self._omega_m = 0.27   # ##
        self.omega_c = 0.2589
        self.omega_DM = 0.23   # ##
        self._omega_L = 0.721  # dark energy density # ##
        self.omega_B = 0.046
        self.omega_re = 9.8e-5
        # ----------------------------------------------------------------------
        self.Delta = 200  # Lopez et al. ---------- M_min=10e-5------  # ##
        self.msq2cmsq = 1e4  # Converts m^2 to cm^2 # ##

    @property
    def omega_m(self):
        return self._omega_m
    @property
    def omega_L(self):
        return self._omega_L

