# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
import numpy as np
# from scipy.stats.stats import _two_sample_transform
from tqdm import tqdm
from config import config
from signal_calc import Signal
from atm_shower import Atm_Shower

from scipy.stats import chi2
from bkgrd_calc import Background
_log = logging.getLogger("pone_dm")


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
        self._t_d = self._find_nearest(self._egrid,
                                       config['simulation paramet' +
                                              'ers']['low enery cutoff'])

        if self.name == 'IceCube':
            self.limit = self.limit_calc_ice
            for i in config['atmospheric showers']['particles of interest']:
                self._bkgrd[i] = np.sum(self._bkgrd[i], axis=0)

        elif self.name == 'POne':
            self.limit = self.limit_calc_POne

        elif self.name == 'combined':
            self.limit = self.limit_calc_combined

    @property
    def limits(self):
        """Returns Calculated Limits for mass grid and SV grd"""
        return self.limit

# Limit calculation ------------------

    def limit_calc_ice(self, mass_grid,
                       sv_grid):

        y = {}
        # for more generations adding the loop ----
        self._signal_grid = np.array([[
                  np.sum((self._signal(self._egrid, mass, sv)),
                         axis=0)[self._t_d:]
                  for mass in mass_grid]
                 for sv in sv_grid]
                 )
        for i in tqdm(config['atmospheric showers']['particles of interest']):
            y[i] = np.array([[chi2.sf(np.sum(
                np.nan_to_num(x**2 /
                              self._bkgrd[i][self._t_d:])), 2)
                            for x in k]
                             for k in self._signal_grid])
        return y, self._signal_grid

    # P-ONE Limit calculation

    def limit_calc_POne(self,
                        mass_grid,
                        sv_grid):

        y = {}
        self._signal_grid = {}
        tmp_dic = {}
        for i in config['atmospheric showers']['particles of interest']:
            self._signal_grid[i] = np.empty((len(sv_grid), len(mass_grid),
                                             len(self._egrid[self._t_d:])))
        tmp_sig_dic = {}
        for _, sv in enumerate(sv_grid):
            for _, mass in enumerate(mass_grid):
                tmp_sig_dic[sv, mass] = (self._signal(self._egrid,
                                         mass, sv))
        for i in config['atmospheric showers']['particles' +
                                               ' of interest']:
            for j, sv in enumerate(sv_grid):
                tmp_dic[i] = []
                for mass in mass_grid:
                    tmp_dic[i].append(tmp_sig_dic[sv, mass][i][self._t_d:])
                self._signal_grid[i][j] = (tmp_dic[i])

        for i in tqdm(config['atmospheric showers']['particles of interest']):
            y[i] = np.array([[chi2.sf(np.sum(
                np.nan_to_num(x**2 /
                              self._bkgrd[i][self._t_d:])), 2)
                            for x in k]
                             for k in self._signal_grid[i]])
        return y, self._signal_grid

    def limit_calc_combined(self, mass_grid, sv_grid):
        y = {}
        self._signal_grid = {}
        tmp_dic = {}
        for i in config['atmospheric showers']['particles of interest']:
            self._signal_grid[i] = np.empty((len(sv_grid), len(mass_grid),
                                             len(self._egrid[self._t_d:])))
        tmp_sig_dic = {}
        for _, sv in enumerate(sv_grid):
            for _, mass in enumerate(mass_grid):
                tmp_sig_dic[sv, mass] = (self._signal(self._egrid,
                                         mass, sv))
        for i in config['atmospheric showers']['particles' +
                                               ' of interest']:
            for j, sv in enumerate(sv_grid):
                tmp_dic[i] = []
                for mass in mass_grid:
                    tmp_dic[i].append(tmp_sig_dic[sv, mass][i][self._t_d:])
                self._signal_grid[i][j] = (tmp_dic[i])

        for i in tqdm(config['atmospheric showers']['particles of interest']):
            y[i] = np.array([[chi2.sf(np.sum(
                np.nan_to_num(x**2 /
                              self._bkgrd[i][self._t_d:])), 2)
                            for x in k]
                             for k in self._signal_grid[i]])

        return y, self._signal_grid

    def _find_nearest(self, array: np.array, value: float):

        """ Returns: index of the nearest vlaue of an array to the given number
        --------------
        idx :  float
        """
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return idx

