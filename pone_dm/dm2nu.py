# -*- coding: utf-8 -*-
# Name: dm2nu.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Collection of methods to calculate the neutrino flux from DM decay

# Imports
import logging
# from numbers import Integral
import numpy as np
import pickle
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from config import config
from constants import pdm_constants
from collections import Counter
_log = logging.getLogger(__name__)


class DM2Nu(object):
    """ Class containing all the necessary methods to convert DM to a surface
    flux at the earth

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        _log.info('Initializing DM to Neutrino methods')
        self._const = pdm_constants()
        self.omega_m = self._const.omega_m
        self.omega_L = self._const.omega_L
        self._d_constructor()
        self.nu_e = pd.read_csv(open('../data/Li_project/nu_e.dat', 'rb'),
                                delim_whitespace=True)
        if config['general']['density'] == 'Burlket':
            self._dphi_de_c = self._dphi_de_c_burkert
        else:
            self._dphi_de_c = self.dphide_channel
        if config['general']["channel"] == 'W':
            # self._channel = "\[Tau]"
            self.m_keys = Counter(self.nu_e['mDM'].values).keys()
            config['simulation parameters']['mass grid'] = (
                [i for i in self.m_keys])

    def galactic_flux(self, E: np.array,
                      m_x: float, sv: float,
                      k: float, J: float):
        """ Fetches the galactic flux
        E : Energy Grid
        m_x : Dark Matter mass
        sv : signma_nu
        k : k factor (majorana: 2 otherwise 4)
        J : J-factor
        """
        return self._dphi_dE_g(
            sv, k, m_x, E, J
        ) * self._dN_nu_E_nu(m_x, E)
        # need to check which one

    def galactic_flux_c(self, E: np.array,
                        m_x: float, sv: float,
                        k: float, J: float):
        """ Fetches the galactic flux
        E : Energy Grid
        m_x : Dark Matter mass
        sv : signma_nu
        k : k factor (majorana: 2 otherwise 4)
        J : J-factor
        """

        e_grid = m_x * 10**self.nu_e[self.nu_e['mDM'] == m_x]['Log[10,x]']
        dNdlogE = self.nu_e[self.nu_e['mDM'] == m_x]["\[Tau]"]
        # phi_nue.append((
        # self.extra_galactic_flux(e_grid, m, 1e-26)) *
        # np.array(dNdlogE) / np.array(e_grid))
        dNdE = np.array(dNdlogE) / np.array(e_grid * np.log(10))
        dNdE = UnivariateSpline(e_grid, dNdE, k=1, s=0, ext=1)

        return self._dphi_dE_g(
            sv, k, m_x, E, J
        ) * dNdE(E)
        # need to check which one

    def extra_galactic_flux_nfw(self, E: np.array,
                                m_x: float, sv: float):
        """ Fetches the extra-galactic flux
        E : Energy grid
        m_x : mass of Dark Matter
        sv : sigma_nu
        """
        return self._dphide_lopez(
            E, m_x, sv
        ) * config["advanced"]["scaling correction"]  # Some error in unit
        # conversion 29.11.21
    # ---------------------------------------------------------------------------
    # Galactic

    def extra_galactic_flux_burkert(self, E: np.array,
                                    m_x: float, sv: float):
        """ Fetches the extra-galactic flux with burkert profile
        E : Energy grid
        m_x : mass of Dark Matter
        sv : sigma_nu
        """
        return self._dphide_burkret(
            E, m_x, sv
        ) * config["advanced"]["scaling correction"]

    def extra_galactic_flux_c(self, E: np.array, m_x: float, snu: float, k=2):
        """Fetches the extra-galactic flux with burkert profile for particluar
           annihilation channel
        E : Energy grid
        m_x : mass of Dark Matter
        sv : sigma_nu
        """
        # the mass factor unaccounted for as of now
        return (self._dphi_de_c(E, m_x, snu, k) *
                config["advanced"]["scaling correction"])  # / m_x**2

    def _dN_nu_E_nu(self, m_x: float, E: np.array):
        """ implements a delta function for the decay

        Parameters
        ----------
        m_x : float
            The DM mass
        E : np.array
            The energies of interest

        Returns
        -------
        np.array
            An array of the same shape as E
        """
        tmp = np.zeros_like(E)
        # Since we work on a grid, the delta function is the closest val
        idE = self._find_nearest(E, m_x) - 1
        tmp[idE] = 2 / E[idE]
        return tmp

    def _dphi_dE_g(self, sigma: float, k: float,
                   m: float, E: np.array, J: float):
        """ The galactic contribution for the neutrino flux

        Parameters
        ----------
        Add

        Returns
        -------
        np.array
        """
        return (
            (1 / (4 * np.pi)) *
            (sigma / (3 * k * m**2)) *
            J * self._dN_nu_E_nu(m, E)
        )

    def _dphi_dE_g_ch(self, sigma: float, k: float,
                      m: float, E: np.array, J: float):
        """ The galactic contribution for the neutrino flux

        Parameters
        ----------
        Add

        Returns
        -------
        np.array
        """
        return (
            (1 / (4 * np.pi)) *
            (sigma / (3 * k * m**2)) *
            J
        ) * config["advanced"]["scaling correction"]

    # ---------------------------------------------------------------------------
    # Extra-Galactic

    def _dN_nu_E_nu_Extra(self, m_x: float, E: np.array):
        """ implements a kinematic cut-off for the decay

        Parameters
        ----------
        m_x : float
            The DM mass
        E : np.array
            The energies of interest

        Returns
        -------
        np.array
            An array of the same shape as E
        """
        return np.array([
            1 / (3 * E) if E < m_x
            else 0
        ])

    def _a_z(self, z: np.array):
        """ Add description
        """
        return 1 / (1+z)

    # TODO: Why is there an H_0 here?
    def _H(self, a: np.array, H_0: float):
        """ time dependent Hubble parameter

        Parameters
        ----------
        a = 1/(1+z)

        Returns
        -------
        H(a)/H_0 normalized
        """
        # The H0 was removed since it cancels later
        return ((self.omega_m / a**3) + self.omega_L)**(1/2)

    def _D_to_inte(self, a: np.array, H_0: float,):
        """ Integrand for D(a)
        """
        return 1 / ((a * self._H(a, H_0))**3)

    def _D(self, a: np.array, H_0: float):
        """ returns:
        D(a(z))
        """

        prefac = 5 * self.omega_m * self._H(a, H_0) / (2)
        integral = np.array([
            quad(self._D_to_inte, 0, a_loc, args=(H_0))[0]
            for a_loc in a
        ])
        return prefac * integral

    def _d_func(self, a: np.array, H_0: float):
        """ Add description
        """
        t = self._D(a, H_0)
        return t / self._D(np.ones_like(a), H_0)

    def _omega_mz(self, z: np.array):
        """ I have neglected omega_re and omega_k couldn't find proper values
        Add description
        """
        a = self._a_z(z)
        return (
            (self.omega_m / a**3) /
            (self.omega_L + (self.omega_m / (a)**3))
        )

    def _sigma_lopez(self, M: float):
        """ returns
        sigma_lopez : float
        """
        return np.exp((2.6 * M**(0.001745)) - 0.2506 * M**0.07536)

    def _f_178(self, M: float, z: np.array):
        """ f_178 from lopez eq b19 : numpy array
        Add description

        """
        A = (
            self._omega_mz(z) *
            (1.907 * (1 + z)**(-3.216) + 0.074)
        )
        al = (
            self._omega_mz(z) *
            (5.907 * (1 + z)**(-3.599) + 2.344)
        )
        beta = (
            self._omega_mz(z) *
            (3.136 * (1 + z)**(-3.068) + 2.349)
        )
        gamma = self._const.gamma
        sigma = (
            self._sigma_lopez(M) *
            self._d(z)
        )
        return (
            A * ((sigma / beta)**(-al) + 1) * np.exp(-gamma / sigma**2)
        )

    def _f_delta(self, M: float, z: np.array):
        """ f_delta Watson et.al. : numpy array
        Add description
        """
        sigma = (
            self._sigma_lopez(M) *
            self._d(z)
        )
        delta = 200  # TODO: Add this to constants
        b_t = (
            np.exp(((delta / 178) - 1) * (0.023 - (0.072 / sigma**2.13))) *
            (delta / 178)**((-0.456 * self._omega_mz(z)) -
                            0.139)
        )
        return self._f_178(M, z) * b_t

    def _g_tild(self, M: float, z: np.array):
        """ For Delta = 200 so g^tilda_200
        returns
        g_tilda : numpy array
        """
        # TODO: All of these constants need to be placed into the constants
        # File
        c_0 = 3.681
        c_1 = 5.033
        al = 6.948
        x_0 = 0.424
        s_0 = 1.047
        s_1 = 1.646
        b = 7.386
        x_1 = 0.526

        A = 2.881
        b = 1.257
        c = 1.022
        d_2 = 0.060

        sigma = (
            self._sigma_lopez(M) *
            self._d(z)
        )

        x = (1 / (1 + z)) * (self.omega_L / self.omega_m)**(1/3)
        # TODO: All of these need descriptions

        def c_min(x):
            return (
                c_0 + (c_1 - c_0) * ((np.arctan(al * (x - x_0)) / np.pi) +
                                     (1/2))
            )

        def s_min(x):
            return (
                s_0 + (s_1 - s_0) * ((np.arctan(b * (x - x_1)) / np.pi) +
                                     (1/2))
            )
        # TODO: This function doesn't seem to be needed

        def B_0(x):
            return c_min(x) / c_min(1.393)

        def B_1(x):
            return s_min(x) / s_min(1.393)

        def s_in(x):
            return B_1(x) * sigma

        def C(x):
            aa = (((s_in(x) / b)**c) + 1)
            dd = np.exp(d_2 / s_in(x)**2)
            return A * aa * dd

        def c_200(x):
            return B_0(x) * C(x)
        c_arr = c_200(x)
        # Removing too high values
        c_arr[c_arr > 100] = 100
        return ((c_arr**3) * (1 - (1 + c_arr)**(-3)) /
                (3 * (np.log(1 + c_arr) - c_arr * (1 + c_arr)**(-1)))**2)

    def _lnsigma_1(self, M: float):
        lnsigma_1 = (0.2506 * (M)**(0.07536)) - (2.6 * (M)**(0.001745))
        return lnsigma_1

    def _dln_sigma_1(self, M: float):
        """returns:
        dln(sigma)/dM : Float
        """
        return (
            0.2506 * 0.07536 * M**(0.07536 - 1) -
            2.6 * 0.001745 * M**(0.001745 - 1)
        )

    def r_delta(self, c_delta, r_0):
        """Returns the upper limit for rho_halo integral
        """
        return c_delta * r_0

    def rho_dm_artsen(self, r):  # 10.06.22-----------------------------------
        """Returns the rho_halo according to the Burkert profile
        """
        delta = 0
        rho_0 = 1.4e7  # [M_sun / kpc^3]
        r_s = 16.1  # kpc
        # rho_local = 0.471  # [GeV / cm^3]
        alpha = 1
        beta = 3
        gamma = 1
        rho_x = rho_0 / (
            ((delta +
             r/r_s)**gamma) * (1 +
                               (r/r_s)**alpha)**((beta-gamma)/alpha))
        return rho_x

    def integral_rho_artsen(self, r_s):
        """Returns the integral over the rho_halo for the burkert
        DM density profile
        """
        r = np.linspace(0.01, self.r_delta(100, r_s), 300)

        rho_dm = self.rho_dm_artsen(r)
        int_rho2 = np.trapz(4 * np.pi * (r**2) * rho_dm**2,
                            x=r, axis=0)

        return int_rho2

    def _G_artsen(self, z):

        def integrand(M):
            return (
                    self._dln_sigma_1(M) *
                    self._f_delta(M, z)
                    )

        aa = (
              (1 / (self._const.omega_DM * self._const.rho_c_mpc *
                    (1+z)**3)**2
               )
              )

        function_vals = np.array([
            integrand(M)
            for M in config["advanced"]["integration grid lopez"]
        ])
        bb = np.trapz(
            function_vals,
            x=config["advanced"]["integration grid lopez"],
            axis=0
        )

        return aa * bb * self.integral_rho_artsen(self._const.r_s)

    def _dphide_artsen(self, E: np.array, m_x: float, snu: float):
        """ returns
        dphi/dE_Lopez : numpy array
        """
        z = m_x / E - 1
        z_tmp = z[z > 0]

        a_G = (
            (1 + self._G_artsen(z_tmp)) *
            (1 + z_tmp)**3
        )

        # multiplide the H_0 ------
        H_z = (self._H(self._a_z(z_tmp), self._const.H_0) *
               self._const.H_0)

        a_g = a_G / H_z
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 8 * np.pi * m_x**2
        res = 2 * aaa * a_g / (3 * E[E < m_x] * b)
        # the factor of 2 for
        # annihiliation to 2 neutrino

        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result

    def _G_lopez(self, z: float):
        """returns
        G_lopez : numpy array
        """
        def integrand(M):
            return (
                    self._dln_sigma_1(M) *
                    self._f_delta(M, z) *
                    self._g_tild(M, z)
                    )

        aa = (
              ((self.omega_m / self._const.omega_DM)**2) *
              self._const.Delta / (3 * self._omega_mz(z)))
        # ------ Here the dNdlogx should be included in the
        # integrand for W, b chanels ----- 19.04.22
        # Using splines to integrate
        function_vals = np.array([
            integrand(M)
            for M in config["advanced"]["integration grid lopez"]
        ])
        bb = np.trapz(
            function_vals,
            x=config["advanced"]["integration grid lopez"],
            axis=0
        )
        # bb = (
        #     quad(integrand, 1e-2, 1e1)[0] +
        #     quad(integrand, 1e1, 1e10)[0] +
        #     quad(integrand, 1e10, 1e17)[0]_dphi_de_c
        # )
        return aa * bb

    def _dphide_lopez(self, E: np.array, m_x: float, snu: float):
        """ returns
        dphi/dE_Lopez : numpy array
        """
        z = m_x / E - 1
        z_tmp = z[z > 0]

        a_G = (
            (1 + self._G_lopez(z_tmp)) *
            (1 + z_tmp)**3
        )

        # multiplide the H_0 ------
        H_z = (self._H(self._a_z(z_tmp), self._const.H_0) *
               self._const.H_0)

        a_g = a_G / H_z
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 8 * np.pi * m_x**2
        res = 2 * aaa * a_g / (3 * E[E < m_x] * b)
        # the factor of 2 for
        # annihiliation to 2 neutrino

        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result  # reason for factor unkown -------

    def rho_dm_burkert(self, r_0, rho_0, r):
        """Returns the rho_halo according to the Burkert profile
        """
        rho_x = rho_0 * (r_0**3) / ((r + r_0) * (r**2 + r_0**2))
        return rho_x

    def Rho_s(self, rho_0, R_0, r_s):
        rho_s = (rho_0 * (R_0 + r_s) * (R_0**2 + r_s**2) /
                 r_s**3)
        return rho_s

    def integral_rho_burkert(self, r_s, rho_0, R_0, r_up):
        """Returns the integral over the rho_halo for the burkert
        DM density profile
        """
        rho_s = self.Rho_s(rho_0, R_0, r_s)
        r = np.linspace(0, self.r_delta(100, r_s), 121)
        rho_dm = self.rho_dm_burkert(r_s, rho_s, r)
        int_rho2 = np.trapz(4 * np.pi * r**2 * rho_dm**2, x=r, axis=0)

        return int_rho2

    def trapz_rho_halo(self):
        """Try trapzoidal method for integral
        """
        return 0

    def _G_burkert(self, z: float):
        """returns
        G_burkert : numpy array
        """
        def integrand(M):
            return (
                    self._dln_sigma_1(M) *
                    self._f_delta(M, z) *
                    self.integral_rho_burkert(self._const.r_s,
                                              self._const.rho_0,
                                              self._const.R_0,
                                              self.r_delta(
                                                  self._const.c_200,
                                                  self._const.r_s))
                                             )

        aa = (
            ((1 / self._const.omega_DM)**2) / (self._const.rho_c_mpc**2) /
            (1+z)**6
        )
        # ------ Here the dNdlogx should be included in the
        # integrand for W, b chanels ----- 19.04.22
        # Using splines to integrate
        function_vals = np.array([
            integrand(M)
            for M in config["advanced"]["integration grid lopez"]
        ])
        bb = np.trapz(
            function_vals,
            x=config["advanced"]["integration grid lopez"],
            axis=0
        )

        return aa * bb

    def _dphide_burkret(self, E: np.array, m_x: float, snu: float):
        """ returns
        dphi/dE_brukert : numpy array
        """
        z = m_x / E - 1
        z_tmp = z[z > 0]
        a_t = (
            (1 + self._G_burkert(z_tmp)) *
            (1 + z_tmp)**3
        )

        # multiplide the H_0 ------
        b_t = (self._H(self._a_z(z_tmp), self._const.H_0) *
               self._const.H_0)

        a_g = a_t / b_t
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 8 * np.pi * m_x**2
        res = 2 * aaa * a_g / (3 * E[E < m_x] * b)
        # the factor of 2 for
        # annihiliation to 2 neutrino

        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result

    def dphide_channel(self, E: np.array, m_x: float, snu: float, k=2):
        """ returns
        dphi/dE_Lopez * dN/dlog(E/m_x)
        """
        # What is the z for T = 1 MeV

        z = m_x / E - 1
        # z = np.linspace(0, self.z_T(), 121)
        e_grid = m_x * 10**self.nu_e[self.nu_e['mDM'] == m_x]['Log[10,x]']
        dNdlogE = self.nu_e[self.nu_e['mDM'] == m_x]["\\[Nu]\\[Mu]"]
        # phi_nue.append((
        # self.extra_galactic_flux(e_grid, m, 1e-26))
        dNdE = np.array(dNdlogE) / np.array(e_grid * np.log(10))
        dNdE = UnivariateSpline(e_grid, dNdE, k=1, s=0, ext=1)
        EW = []
        for i, e in enumerate(E):
            if i == 0:
                EW.append(e)
            else:
                EW.append(e - E[i-1])
        EW = np.array(EW)

        def a_t(z_):
            # multiplide the H_0 ------
            b_t = (self._H(self._a_z(z_), self._const.H_0) *
                   self._const.H_0)
            return ((1 + self._G_lopez(z_)) *
                    (1 + z_)**3 / b_t)
        a_g = []
        for i, Z in enumerate(z):
            if Z <= 0:
                for j in z[i:]:
                    a_g.append(0)
                break
            else:
                tmp_a_g = a_t(z[:i])
            # a_g.append(np.trapz(tmp_a_g, z[:i]))
                a_g.append(np.dot(tmp_a_g * dNdE(E[i]) / (1 + z[:i])**2,
                           # redshift factor from DM for spectrum
                                  m_x / EW[:i] - 1))
        a_g = np.array(a_g)
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 4 * k * np.pi * m_x**2
        res = aaa * a_g / (b)
        # the factor of 2 for
        # annihiliation to 2 neutrino

        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result

    def _dphi_de_c_burkert(self, E: np.array, m_x: float, snu: float, k=2):
        """ returns
        dphi/dE_brukert : numpy array
        """
        z = m_x / E - 1
        # z = np.linspace(0, self.z_T(), 121)
        e_grid = m_x * 10**self.nu_e[self.nu_e['mDM'] == m_x]['Log[10,x]']
        dNdlogE = self.nu_e[self.nu_e['mDM'] == m_x]["\\[Nu]\\[Mu]"]
        # phi_nue.append((
        # self.extra_galactic_flux(e_grid, m, 1e-26))
        dNdE = np.array(dNdlogE) / np.array(e_grid * np.log(10))
        dNdE = UnivariateSpline(e_grid, dNdE, k=1, s=0, ext=1)
        EW = []
        for i, e in enumerate(E):
            if i == 0:
                EW.append(e)
            else:
                EW.append(e - E[i-1])
        EW = np.array(EW)

        def a_t(z_):
            # multiplide the H_0 ------
            b_t = (self._H(self._a_z(z_), self._const.H_0) *
                   self._const.H_0)
            return ((1 + self._G_burkert(z_)) *
                    (1 + z_)**3 / b_t)
        a_g = []
        for i, Z in enumerate(z):
            if Z <= 0:
                for j in z[i:]:
                    a_g.append(0)
                break
            else:
                tmp_a_g = a_t(z[:i])
            # a_g.append(np.trapz(tmp_a_g, z[:i]))
                a_g.append(np.dot(tmp_a_g * dNdE(E[i]) / (1+z)**2,
                           # as Li Suggested for redshift factor
                                  m_x / EW[:i] - 1))
        a_g = np.array(a_g)
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 4 * k * np.pi * m_x**2
        res = aaa * a_g / b
        # the factor of 2 for
        # annihiliation to 2 neutrino

        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result

    def z_T(self):
        T0 = 2.725  # in Kelvin , From Mather et. al 1999 (saw in Luzzi)
        # In standard Model (1+z)^(1 + alpha), alpha=0 !!!!!
        # in Luzzi they found alpha =
        T_MeV_K = 11604525006.1657
        return (T_MeV_K / (T0)) - 1

    def _find_nearest(self, array, value):
        """ Add description
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _d_constructor(self):
        """ Constructs tables for _d to avoid costly intgeration

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        load_str = config["advanced"]["_d storage"] + "d_store.p"
        try:
            _log.debug("Loading pickle file for d spline")
            _log.debug("Searching for " + load_str)
            loaded_d = pickle.load(open(load_str, "rb"))
            self._d = loaded_d[1]
        except FileNotFoundError:
            _log.debug("Failed to load d data")
            _log.debug("Constructing it, this may take a while")
            z_grid = config["advanced"]["construction grid _d"]
            a_grid = self._a_z(z_grid)
            d_grid = self._d_func(a_grid,
                                  self._const.H_0)
            self._d = UnivariateSpline(z_grid,
                                       d_grid, k=1, s=0, ext=3)
            _log.debug("Finished construction")
            _log.debug("Dumping the data")
            pickle.dump([d_grid, self._d],
                        open(load_str, "wb"))
