# -*- coding: utf-8 -*-
# Name: signal_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the signal counts for DM -> Neutrino

# Imports
import logging
import numpy as np
from config import config
from pone_aeff import Aeff
from dm2nu import DM2Nu
from constants import pdm_constants
from detectors import Detector
_log = logging.getLogger("pone_dm")


class Signal(object):
    """ Class of methods to calculate signal at detector from DM -> Neutrinos

    Parameters
    ----------
    aeff: Aeff
        An Aeff object
    dm_nu: DM2Nu
        A DM2Nu object
    detector: Detector
        A Detector object
    """
    def __init__(self, aeff: Aeff, dmnu: DM2Nu,
                 detector: Detector, year=config['general']['year']):
        self._aeff = aeff

        self._dmnu = dmnu
        self._detector = detector
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']
        self._year = year
        self._ewidth = self._aeff.ewidth
        self._egrid = self._aeff.egrid

        name = config['general']['detector']
        if name == 'IceCube':
            print(name)
            self._signal_calc = self._signal_calc_ice
        elif name == 'POne':
            print(name)
            self._signal_calc = self._signal_calc_pone

    @property
    def signal_calc(self):
        """Returns appropriate signal calculation function
        total_counts : np.array
        """
        return self._signal_calc

    def _signal_calc_ice(self, egrid: np.array, mass: float,
                         sv: float):
        """ Calculates the expected signal given the mass, sigma*v and angle

        Parameters
        ----------
        egrid : np.array
            The energy grid to calculate on
        mass : float
            The DM mass
        sv : float
            The sigma * v of interest
        angle_grid : np.array
            The incoming angles

        Returns
        -------
        total_new_counts : np.array (len(year),len(E_grid))
            The total new counts
        """
        # Extra galactic
        _extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)

        # Galactic
        total_new_counts = []
        # TODO: Need to configure for IceCube ------

        _ours = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d + self._const.J_p + self._const.J_s
        )
        # Converting fluxes into counts with effective area of IceCube !!!!
        #  These steps take a lot of time !!!!
        total_flux = _ours+_extra
        for y in self._year:
            total_new_counts.append(self._detector.sim2dec(total_flux, y)[
                'numu'] / config["advanced"]["scaling correction"])
        # the sim_to_dec omits the dict but we assume
        # same for all neutrino flavours

        return total_new_counts

    def _signal_calc_pone(self, egrid: np.array, mass: float,
                          sv: float):
        """ Calculates the expected signal given the mass, sigma*v and angle

        Parameters
        ----------
        egrid : np.array
            The energy grid to calculate on
        mass : float
            The DM mass
        sv : float
            The sigma * v of interest
        angle_grid : np.array
            The incoming angles

        Returns
        -------
        total_new_counts : np.array
            The total new counts
        """

        # Extra galactic

        extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)
        _flux = {}
        # Galactic

        _flux[15] = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d1 + self._const.J_p1 + self._const.J_s1
        )

        _flux[85] = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d2 + self._const.J_p2 + self._const.J_s2
        )

        _flux[120] = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d3 + self._const.J_p3 + self._const.J_s3
        )
        for i in _flux.keys():
            _flux[i] = np.array(_flux[i])+np.array(extra)

        total_counts = self._detector.sim2dec(_flux, boolean_sig=True)[
            'numu'] / config["advanced"]["scaling correction"]

        return total_counts

    def _find_nearest(self, array: np.array, value: float):

        """ Returns: index of the nearest vlaue of an array to the given number
        --------------
        idx :  float
        """
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return idx
