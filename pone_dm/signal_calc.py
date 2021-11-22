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
                 detector: Detector):
        self._aeff = aeff

        self._dmnu = dmnu
        self._detector = detector
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']
        self._year = config['general']['year']
        self._ewidth = self._aeff.ewidth
        self._egrid = self._aeff.egrid

        name = config['general']['detector']
        if name == 'IceCube':
            self._signal_calc = self._signal_calc_ice
        elif name == 'POne':
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
        total_new_counts : np.array
            The total new counts
        totla_flux : np.array
            the total_flux
        """
        # Extra galactic
        _extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)

        # Galactic

        # TODO: Need to configure for IceCube ------

        _ours = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d + self._const.J_p + self._const.J_s
        )
        # Converting fluxes into counts with effective area of IceCube !!!!
        #  These steps take a lot of time !!!!
        total_flux = _ours+_extra
        total_new_counts = self._detector.sim2dec(total_flux, self._year)[0][
            'numu'] / config["advanced"]["scaling correction"]
        # the sim_to_dec omits the dict but we assume
        # same for all neutrino flavours

        return total_new_counts

    def _signal_calc_pone(self, egrid: np.array, mass: float,
                          sv: float, angle_grid: np.array):
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
        self._extra_flux = extra

        # Galactic

        ours_15 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d1 + self._const.J_p1 + self._const.J_s1
        )

        ours_55 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d2 + self._const.J_p2 + self._const.J_s2
        )

        ours_51 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d3 + self._const.J_p3 + self._const.J_s3
        )

        # Convolving

        down_angles = []
        horizon_angles = []
        extra_down = []
        extra_hor = []
        ours_down = []
        ours_hor = []
        if config["general"]["detector"] == "POne":
            for angle in angle_grid:
                rad = np.deg2rad(np.abs(angle - 90.))

                # Downgoing
                if np.pi / 3 <= rad <= np.pi / 2:
                    down_angles.append(rad)
                    extra_down.append(
                        extra * self._uptime *
                        self._ewidth * self._aeff.spl_A15(self._egrid)
                    )
                    ours_down.append(
                        ours_15 * self._uptime *
                        self._ewidth * self._aeff.spl_A15(self._egrid)
                    )

                # Horizon
                else:
                    horizon_angles.append(rad)
                    extra_hor.append(
                        extra * self._uptime *
                        self._ewidth * self._aeff.spl_A55(self._egrid)
                    )
                    ours_hor.append(
                        ours_55 * self._uptime *
                        self._ewidth * self._aeff.spl_A55(self._egrid)
                    )

            # Converting to numpy arrays

            extra_down = np.array(extra_down)
            extra_hor = np.array(extra_hor)
            ours_down = np.array(ours_down)
            ours_hor = np.array(ours_hor)
            down_angles = np.array(down_angles)
            horizon_angles = np.array(horizon_angles)

            # Integrating
            # Extra

            self._extra = np.zeros_like(self._egrid)

            # Downgoing

            sorted_ids = np.argsort(down_angles)
            self._extra += np.trapz(extra_down[sorted_ids],
                                    x=down_angles[sorted_ids], axis=0)

            # Horizon we assume it is mirrored

            sorted_ids = np.argsort(horizon_angles)
            self._extra += 2. * np.trapz(extra_hor[sorted_ids],
                                         x=horizon_angles[sorted_ids], axis=0)

            # Upgoing we assume the same flux for all
            self._extra += (
                (np.pi / 2 - np.pi / 3) *
                extra * self._uptime *
                self._ewidth * self._aeff.spl_A51(self._egrid)
            )

            # Ours
            self._ours = np.zeros_like(self._egrid)

            # Downgoing
            sorted_ids = np.argsort(down_angles)
            self._ours += np.trapz(ours_down[sorted_ids],
                                   x=down_angles[sorted_ids], axis=0)
            # Horizon we assume it is mirrored
            sorted_ids = np.argsort(horizon_angles)
            self._ours += 2. * np.trapz(ours_hor[sorted_ids],
                                        x=horizon_angles[sorted_ids], axis=0)
            # Upgoing we assume the same flux for all
            self._ours += (
                (np.pi / 2 - np.pi / 3) *
                ours_51 * self._uptime *
                self._ewidth * self._aeff.spl_A51(self._egrid)
            )
            total_new_counts = (
                (self._extra + self._ours) /
                config["advanced"]["scaling correction"]  # Some unknown error
            )

        return total_new_counts

    def _find_nearest(self, array: np.array, value: float):

        """ Returns: index of the nearest vlaue of an array to the given number
        --------------
        idx :  float
        """
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return idx
