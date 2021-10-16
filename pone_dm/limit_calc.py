# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
import numpy as np
from tqdm import tqdm
from config import config
from signal_calc import Signal
from atm_shower import Atm_Shower

from scipy.stats import chi2
from bkgrd_calc import Background
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
    def __init__(self, signal: Signal,
                 shower_sim: Atm_Shower,
                 background: Background):
        _log.info('Initializing the Limits object')

        self._shower = shower_sim
        self._sig = signal

        self._egrid = self._shower.egrid

        self._background = background
        self._massgrid = config["simulation parameters"]["mass grid"]
        self._svgrid = config["simulation parameters"]["sv grid"]
        self._uptime = config['simulation parameters']['uptime']
        _log.info('Preliminary calculations')
        _log.debug('The total atmospheric flux')
        self._year = config['general']['year']
        self.name = config['general']['detector']
        self._bkgrd = self._background.bkgrd
        self._signal = self._sig._signal_calc
        if self.name == 'IceCube':
            self.limit = self.limit_calc_ice(self._massgrid, self._svgrid)

        elif self.name == 'POne':
            self.limit = self.limit_calc_pone(self._massgrid, self._svgrid)

    @property
    def limits(self):
        """Returns Calculated Limits for mass grid and SV grd"""
        return self.limit

# Limit calculation for IceCube------------------

    def limit_calc_ice(self,
                       mass_grid,
                       sv_grid):

        y = {}
        # for more generations adding the loop ----

        for i in tqdm(config['atmospheric showers']['particles of interest']):

            _log.info('Starting the limit calculation for IceCube detector')

            # The low energy cut off
            self._t_d = self._find_nearest(self._egrid, 5e2)

            self._limit_scan_grid_base = np.array(
                [[
                  ((self._signal(self._egrid, mass,
                    sv)**2.
                    )[self._t_d:] /
                   self._bkgrd[i][self._t_d:])
                  for mass in mass_grid]
                 for sv in sv_grid]
                 )

            y[i] = np.array([[
                              chi2.sf(np.sum(np.nan_to_num(x)), 2)
                            for x in k] for k in self._limit_scan_grid_base
            ])
        return y

# Limit calculation for Pone----------------------

    def limit_calc_pone(self,
                        mass_grid,
                        sv_grid
                        ):
        """ Scans the masses and sigma*nu and calculates
        the corresponding limits

        Parameters
        ----------
        mass_grid : np.array
            The masses to scan
        sv_grid : np.array
            The sigma * v grid

        Returns
        -------
        list
            The resulting chi values
        """

        _log.info('Starting the limit calculation')
        # The low energy cut off
        y = {}

        for i in (config['atmospheric showers']['particles of interest']):

            self._t_d = self._find_nearest(self._egrid, 5e2)

            self._limit_grid = np.array([[
                            (self._signal(
                                self._egrid, mass,
                                sv,
                                config['atmospheric showers']['theta angles']
                                )**2.
                             )[self._t_d:] /
                            self._bkgrd[i][self._t_d:]
                            for mass in mass_grid]
                            for sv in tqdm(sv_grid)
                            ])
            y[i] = np.array([[chi2.sf(np.sum(np.nan_to_num(x)), 2)
                            for x in k] for k in self._limit_grid])
        return y

    def _find_nearest(self, array: np.array, value: float):

        """ Returns: index of the nearest vlaue of an array to the given number
        --------------
        idx :  float
        """
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return idx
