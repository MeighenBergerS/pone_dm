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
import pickle
import os
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
        self.particles = config['atmospheric showers']['particles of interest']
        self._bkgrd = self._background.bkgrd
        self._signal = self._sig._signal_calc
        self._t_d = self._find_nearest(self._egrid,
                                       config['simulation paramet' +
                                              'ers']['low enery cutoff'])

        if self.name == 'IceCube':
            self.limit = self.limit_calc_ice
            for i in self.particles:
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
        try:
            _log.info('Fetching precalculated signal grid for IceCube')
            self._signal_grid = pickle.load(open(
                            '../data/limits_signal_IceCube.pkl', 'rb'))
        except FileNotFoundError:
            _log.info('No precalculated signal grid found')
            _log.info('Calculating the signal grid for IceCube')
            # for more generations adding the loop ----
            self._signal_grid = np.array([[
                     np.sum(self._signal(self._egrid, mass, sv),
                            axis=0)[self._t_d:]
                     for mass in mass_grid]
                     for sv in sv_grid]
                     )
        for i in tqdm(self.particles):
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
        for i in self.particles:
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

        for i in tqdm(self.particles):
            y[i] = np.array([[chi2.sf(np.sum(
                np.nan_to_num(x**2 /
                              self._bkgrd[i][self._t_d:])), 2)
                            for x in k]
                             for k in self._signal_grid[i]])
        # Dumping the signal_grid
        pickle.dump(self._signal_grid, open(
            '../data/limits_signal_POne.pkl', "wb"
        ))
        return y, self._signal_grid

    def limit_calc_combined(self, mass_grid, sv_grid):
        y = {}
        try:
            _log.info('Fetching combined signal_grid')
            self._signal_grid = pickle.load(
                open("../data/limits_signal_combined.pkl", "rb")
            )
        except FileNotFoundError:
            self._signal_grid = {}
            _log.info('Fetching precalculated signal grids P-ONE and IceCube')
            pone_signal_bool = os.path.isfile(
                '../data/limits_signal_POne.pkl')
            ice_signal_bool = os.path.isfile(
                '../data/limits_signal_IceCube.pkl')
            if ice_signal_bool is True and pone_signal_bool is True:
                self._signal_grid_pone = pickle.load(open(
                    '../data/limits_signal_POne.pkl', 'rb'
                ))
                self._signal_grid_ice = pickle.load(open(
                    '../data/limits_signal_IceCube.pkl', 'rb'
                ))
                for i in self.particles:
                    self._signal_grid[i] = np.add(self._signal_grid_pone[i],
                                                  self._signal_grid_ice)
            elif ice_signal_bool is True and pone_signal_bool is False:
                self._signal_grid_ice = pickle.load(open(
                    '../data/limits_signal_IceCube.pkl', 'rb'
                ))
                print('type of signal grid IceCube' +
                      str(type(self._signal_grid_ice)))
                print(self.name)
                _log.info('Calculating the P-ONE signal grid')
                self._signal_grid_pone = {}
                tmp_dic = {}
                for i in self.particles:
                    self._signal_grid_pone[i] = np.empty(
                        (len(sv_grid), len(mass_grid),
                         len(self._egrid[self._t_d:])))
                tmp_sig_dic = {}
                for _, sv in enumerate(sv_grid):
                    for _, mass in enumerate(mass_grid):
                        tmp_sig_dic[sv, mass] = (
                            self._sig.signal_calc_pone(
                                                self._egrid,
                                                mass, sv))
                for i in config['atmospheric showers']['particles' +
                                                       ' of interest']:
                    for j, sv in enumerate(sv_grid):
                        tmp_dic[i] = []
                        for mass in mass_grid:
                            tmp_dic[i].append(
                                tmp_sig_dic[sv, mass][i][self._t_d:])
                        self._signal_grid_pone[i][j] = (tmp_dic[i])
                for i in self.particles:
                    self._signal_grid[i] = np.add(
                                        self._signal_grid_pone[i],
                                        self._signal_grid_ice)
                pickle.dump(self._signal_grid_pone, open(
                    '../data/limists_signal_POne.pkl', 'wb'
                ))
                pickle.dump(self._signal_grid, open(
                   '../data/limits_signal_combined.pkl', 'wb'
                ))
            elif pone_signal_bool is True and ice_signal_bool is False:
                self._signal_grid_pone = pickle.load(open(
                    '../data/limits_signal_POne.pkl', 'rb'
                ))
                _log.info('calculating IceCube signal grid')
                _log.info('No precalculated signal grid found')
                # for more generations adding the loop ---
                self._signal_grid_ice = np.array([[
                         np.sum(self._sig.signal_calc_ice(
                             self._egrid, mass, sv),
                                axis=0)[self._t_d:]
                         for mass in mass_grid]
                         for sv in sv_grid]
                         )
                for i in self.particles:
                    self._signal_grid[i] = np.add(
                        self._signal_grid_pone[i],
                        self._signal_grid_ice)
                pickle.dump(self._signal_grid_ice, open(
                    '../data/limists_signal_IceCube.pkl', 'wb'
                ))
                pickle.dump(self._signal_grid, open(
                   '../data/limits_signal_combined.pkl', 'wb'
                ))
            elif pone_signal_bool is False and ice_signal_bool is False:
                print('both false')
                tmp_dic = {}
                for i in self.particles:
                    self._signal_grid[i] = np.empty(
                        (len(sv_grid), len(mass_grid),
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
                            tmp_dic[i].append(
                                tmp_sig_dic[sv, mass][i][self._t_d:])
                        self._signal_grid[i][j] = (tmp_dic[i])
            pickle.dump(self._signal_grid, open(
               '../data/limits_signal_combined.pkl', 'wb'
               ))

        for i in tqdm(self.particles):
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
