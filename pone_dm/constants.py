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
        # Galactic centre
        # thermally averaged cross section to neutrinos
        # 300 GeV to 3 TeV from Fermi-LAT and HESS
        self.Kappa_m = 2  # majorana DM  ----- This is used for the paper-----
        self.Kappa_d = 4  # Dirac DM

        self.h = 0.674

        self.n = -1.316  # power spetcrum index
        self.rho_c_mpc = 2.7754e11  # h^-2 M_0 Mpc^-3-------------> Mpc!!!!!

        self.gamma = 1.3186  # NFW parameter --- slope parameter -----
        self.rho_m = 29.65e12  # M_sun Mpc^-3
        # ----------------------------------------------------------------------
        # P-ONE
        # values of  J for P-ONE ------------

        # c_zenith_1= [-1,-0.5]
        self.J_s1 = 0.87e23
        self.J_p1 = 0.85e17
        self.J_d1 = 1.4e11
        # c_zenith_2= [-0.5,0.5]
        self.J_s2 = 1.2e23
        self.J_p2 = 1.2e17
        self.J_d2 = 2.0e11

        # c_zenith_3= [0.5,1]
        self.J_s3 = 0.13e23
        self.J_p3 = 0.13e17
        self.J_d3 = 0.18e11

        # Total J
        self.J = (
            self.J_s1 + self.J_p1 + self.J_d1 +
            self.J_s2 + self.J_s3 + self.J_p2 +
            self.J_p3 + self.J_d2 + self.J_d3
        )
        # ----------------------------------------------------------------------
        # Extra-Galactic
        self.M = np.logspace(-1, 15, 400, 10)
        self.E = np.logspace(-1, 8, 100, 10)
        self.Z = np.linspace(0, 60, 60)

        self.H_0 = 71  # km/(Mpc*s) hubble time ---
        # rho_s scale density
        self.r_s = 20  # kpc scale radius ----
        self.rho_0 = 0.4  # GeV cm^(-3)
        self.R_0 = 8.127  # kpc
        self.omega_k = -0.09
        self.omega_m = 0.27
        self.omega_c = 0.2589
        self.omega_DM = 0.23
        self.omega_L = 0.721  # dark energy density
        self.omega_B = 0.046
        self.omega_re = 9.8e-5
        self.rho_B = self.omega_B * self.rho_c_mpc  # baryonic mass density
        # radiation density of these the most accurately measured ---------
        # ----------------------------------------------------------------------
        # DM
        self.m_dm = np.array([1, 1e2, 1e4, 1e6])
        self.k = 2
        self.sigma_nu = 5e-29  # cm^3 s
        # ----------------------------------------------------------------------
        # Not sure where to put these
        self.Delta = 200  # Lopez et al. ---------- M_min=10e-5------
        self.msq2cmsq = 1e4  # Converts m^2 to cm^2
