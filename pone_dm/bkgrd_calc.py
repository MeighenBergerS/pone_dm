# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
import numpy as np
from config import config
from pone_aeff import Aeff
from dm2nu import DM2Nu
from atm_shower import Atm_Shower
from constants import pdm_constants
from IceCube_extraction import Icecube_data
_log = logging.getLogger(__name__)


class background(object):
    """
    Class to calculate background counts for both P-One and IceCube detectors
    """
    def __init__(self, aeff: Aeff, dm_nu: DM2Nu, shower_sim: Atm_Shower):
        self.icecube_eff = Icecube_data
        self._aeff = aeff
        self._dmnu = dm_nu
        self._shower = shower_sim
        self._egrid = self._shower.egrid
        self._ewidth = self._shower.ewidth
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']

        if config["general"]["detector"] == "POne":
            _log.info('Initializing the Limits object')

            _log.info('Preliminary calculations')
            _log.debug('The total atmospheric flux')

            # The fluxes convolved with the effective area
            self.bkgrd_down = {}

            self.bkgrd_horizon = {}

            # backgorund dictionary repositioned
            self._bkgrd = {}

            for i in (config['atmospheric showers']['particles of interest']):
                self.bkgrd_down[i] = []
                down_angles = []
                self.bkgrd_horizon[i] = []
                horizon_angles = []
            # Astrophysical is the same everywhere
                for angle in config['atmospheric showers']['theta angles']:
                    rad = np.deg2rad(np.abs(angle - 90.))

                    # Downgoing
                    if np.pi / 3 <= rad <= np.pi / 2:

                        down_angles.append(rad)
                        self.bkgrd_down[i].append(
                            (self._shower.flux_results[angle][i] +
                             self._dphi_astro(self._egrid)) * self._uptime *
                            self._ewidth * self._aeff.spl_A15(self._egrid)
                        )

                    # Horizon
                    else:
                        horizon_angles.append(rad)
                        self.bkgrd_horizon[i].append(
                            (self._shower.flux_results[angle][i] +
                             self._dphi_astro(self._egrid)) * self._uptime *
                            self._ewidth * self._aeff.spl_A55(self._egrid)
                        )

                # Converting to numpy arrays
                self.bkgrd_down[i] = np.array(self.bkgrd_down[i])
                self.bkgrd_horizon[i] = np.array(self.bkgrd_horizon[i])
                down_angles = np.array(down_angles)
                horizon_angles = np.array(horizon_angles)

                # Integrating

                self._bkgrd[i] = np.zeros_like(self._egrid)
                sorted_ids = np.argsort(down_angles)

                # Downgoing
                self._bkgrd[i] += np.trapz(self.bkgrd_down[i][sorted_ids],
                                           x=down_angles[sorted_ids], axis=0)

                # Horizon we assume it is mirrored
                sorted_ids = np.argsort(horizon_angles)
                self._bkgrd[i] += 2. * np.trapz(self.bkgrd_horizon[i][
                                                        sorted_ids],
                                                x=horizon_angles[sorted_ids],
                                                axis=0)

                # Upgoing we assume the same flux for all
                self._bkgrd[i] += (
                    (np.pi / 2 - np.pi / 3) *
                    (self._shower.flux_results[0.][i] +
                     self._dphi_astro(self._egrid)) * self._uptime *
                    self._ewidth * self._aeff.spl_A51(self._egrid)
                )
                self._bkgrd[i] = self._bkgrd[i] * 46

        elif config["general"]["detector"] == "IceCube":
            self._bkgrd = self.Flux_eff
            print("IceCube Detector")  # -------  ------    06.09.21

    def Flux_eff(self):  # ----- 06.09.21 -------
        total_bkgrd = {}
        for i in config["atmospheric showers"]["particles of interest"]:
            for year in range(10):
                for theta in config["atmospheric showers"]["thetas"]:
                    total_bkgrd[i, year, theta] = self.icecube_eff.flux_results()[year, theta] * self.icecube_eff.spl_e_areas[year](self.egrid)

        return total_bkgrd

    @property
    def egrid(self):
        """ Fetches the calculation egrid

        Returns
        -------
        np.array
            The energy grid
        """
        return self._egrid

    @property
    def ewidth(self):
        """ Fetches the calculation e widths

        Returns
        -------
        np.array
            The energy grid widths
        """
        return self._ewidth

    def _find_nearest(self, array, value):
        """ Add description
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _dphi_astro(self, E):
        """
        Astrophysical flux because of muon background as per the power
        law described in https://arxiv.org/pdf/1907.11266.pdf

        Add description
        """
        return 1e-18 * 1.66 * ((E/1e5)**(-2.53))
        # 1e-18 * 6.45 * (E / 1e5)**(-2.89)

    def atmos_flux(self):
        return self._bkgrd


