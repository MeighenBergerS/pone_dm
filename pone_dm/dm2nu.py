# -*- coding: utf-8 -*-
# Name: dm2nu.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Collection of methods to calculate the neutrino flux from DM decay

# Imports
import logging
import numpy as np
import pickle
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from config import config
from constants import pdm_constants

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
        self.nu_e = pd.read_csv(open('../data/nu_e.dat', 'rb'),
                                delim_whitespace=True)
        self._dphi_de_c = self.dphide_channel
        self.channel = config['general']["channel"]
        if config['general']["channel"] != 'All':
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
        ) * config["advanced"]["scaling correction"]  # TODO: Unit correction need to check which one

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
        ch = self.channel
        e_grid = m_x * 10**self.nu_e[self.nu_e['mDM'] == m_x]['Log[10,x]']
        dNdlogE = self.nu_e[self.nu_e['mDM'] == m_x][ch]
        # phi_nue.append((
        # self.extra_galactic_flux(e_grid, m, 1e-26)) *
        # np.array(dNdlogE) / np.array(e_grid))
        dNdE = np.array(dNdlogE) / np.array(e_grid * np.log(10))
        dNdE = UnivariateSpline(e_grid, dNdE, k=1, s=0, ext=1)

        return self._dphi_dE_g(
            sv, k, m_x, E, J
        ) * dNdE(E)

    def extra_galactic_flux(self, E: np.array,
                            m_x: float, sv: float):
        """ Fetches the extra-galactic flux
        E : Energy grid
        m_x : mass of Dark Matter
        sv : sigma_nu
        """
        return self._dphide_lopez(
            E, m_x, sv
        ) * config["advanced"]["scaling correction"]  # Some error in unit conversion 29.11.21
    # ---------------------------------------------------------------------------
    # Galactic

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
        tmp[idE] = 1 / E[idE]
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
            (self._dN_nu_E_nu(m, E)) * J
        )
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
        """ f_delta from lopez eq b21 : numpy array
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
        a_arr = c_200(x)
        # Removing too high values
        a_arr[a_arr > 100] = 100
        return ((a_arr**3) * (1 - (1 + a_arr)**(-3)) /
                (3 * (np.log(1 + a_arr) - a_arr * (1 + a_arr)**(-1)))**2)

    def _dln_sigma_1(self, M: float):
        """returns:
        dln(sigma)/dM : Float
        """
        return (
            0.2506 * 0.07536 * M**(0.07536 - 1) -
            2.6 * 0.001745 * M**(0.001745 - 1)
        )

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
            self._const.Delta / (3 * self._omega_mz(z))
        )
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
        #     quad(integrand, 1e10, 1e17)[0]
        # )
        return aa * bb

    def _dphide_lopez(self, E: np.array, m_x: float, snu: float):
        """ returns
        dphi/dE_Lopez : numpy array
        """
        z = m_x / E - 1
        z_tmp = z[z > 0]
        a_t = (
            (1 + self._G_lopez(z_tmp)) *
            (1 + z_tmp)**3
        )

        # multiplide the H_0 ------
        b_t = (self._H(self._a_z(z_tmp), self._const.H_0) *
               self._const.H_0)

        a_g = a_t / b_t
        aaa = snu * (self._const.omega_DM * self._const.rho_c_mpc)**2
        b = 8 * np.pi * m_x**2
        res = aaa * a_g / (3 * E[E < m_x] * b)
        # Padding with zeros
        result = np.zeros_like(E)
        result[0:len(res)] = res
        return result

# Cook book -------------------------------------------------------------

    def dphide_channel(self, E: np.array, m_x: float, snu: float, k=2):
        """ returns
        dphi/dE_Lopez * dN/dlog(E/m_x)
        """
        # What is the z for T = 1 MeV
        ch = self.channel
        z = m_x / E - 1
        # z = np.linspace(0, self.z_T(), 121)
        e_grid = m_x * 10**self.nu_e[self.nu_e['mDM'] == m_x]['Log[10,x]']
        dNdlogE = self.nu_e[self.nu_e['mDM'] == m_x][ch]
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

