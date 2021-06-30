# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
import numpy as np
from .config import config
from .pone_aeff import Aeff
from .dm2nu import DM2Nu
from .atm_shower import Atm_Shower

_log = logging.getLogger(__name__)

class Limits(object):
    """ Collection of methods to calculate the limits

    Parameters
    ----------
    aeff: Aeff
        An Aeff object
    dm_nu: DM2Nu
        A DM2Nu object
    shower_sim: Atm_Shower
        A Atm_Shower object
    """
    def __init__(self, aeff: Aeff, dm_nu: DM2Nu, shower_sim: Atm_Shower):
        _log.info('Initializing the Limits object')
        self._aeff = aeff
        self._dmnu = dm_nu
        self._shower = shower_sim
        self._egrid = self._shower.egrid
        self._ewidth = self._shower.ewidth
        self._uptime = config['simulation parameters']['uptime']
        _log.info('Preliminary calculations')
        _log.debug('The total atmospheric flux')
        # The fluxes convolved with the effective area
        self._numu_bkgrd_down = []
        down_angles = []
        self._numu_bkgrd_horizon = []
        horizon_angles = []
        for angle in config['atmospheric showers']['theta angles']:
            rad = np.deg2rad(np.abs(angle - 90.))
            # Downgoing
            if np.pi / 3 <= rad <= np.pi / 2:
                down_angles.append(rad)
                self._numu_bkgrd_down.append(
                    self._shower[angle]['numu'] * self._uptime *
                    self._ewidth * self._aeff.spl_A15(self._egrid)
                )
            # Horizon
            else:
                horizon_angles.append(rad)
                self._numu_bkgrd_horizon.append(
                    self._shower[angle]['numu'] * self._uptime *
                    self._ewidth * self._aeff.spl_A55(self._egrid)
                )
        # Converting to numpy arrays
        self._numu_bkgrd_down = np.array(self._numu_bkgrd_down)
        self._numu_bkgrd_horizon = np.array(self._numu_bkgrd_horizon)

        # Integrating
        self._numu_bkgrd = np.zeros_like(self._egrid)
        # Downgoing
        self._numu_bkgrd += np.trapz(self._numu_bkgrd_down,
                                     x=down_angles, axis=0)
        # Horizon we assume it is mirrored
        self._numu_bkgrd += 2. * np.trapz(self._numu_bkgrd_horizon,
                                          x=horizon_angles, axis=0)
        # Upgoing we assume the same flux for all
        self._numu_bkgrd += (
            (np.pi / 2 - np.pi / 3) *
            self._shower[0.]['numu'] * self._uptime *
            self._ewidth * self._aeff.spl_A51(self._egrid)
        )

    def chi
