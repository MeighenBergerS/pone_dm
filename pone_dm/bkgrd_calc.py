# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
from config import config
# from tqdm import tqdm
from atm_shower import Atm_Shower
from detectors import Detector
import pickle
import csv
import numpy as np
_log = logging.getLogger("pone_dm")


class Background(object):
    """
    Class to calculate background counts for both P-One and IceCube detectors
    """
    def __init__(self, shower_sim: Atm_Shower, detector: Detector,
                 year=config['general']['year']):

        self._year = year
        self._detector = detector
        self._shower = shower_sim
        self._uptime = config['simulation parameters']['uptime']
        self.name = config['general']['detector']
        _log.info('Initializing the Limits object')
        _log.info('Preliminary calculations')
        _log.debug('The total atmospheric flux')
        self._pone_smearing = config['pone']['smearing']
        if self._pone_smearing == 'smeared':
            self._smb = True
        elif self._pone_smearing == 'unsmeared':
            self._smb = False
        # Check this again
        if self.name == "IceCube":
            self.days = 60. * 24
            self.minutes = 60.
            self._ice_data = [
                        '../data/icecube_10year_ps/events/IC40_exp.csv',
                        '../data/icecube_10year_ps/events/IC59_exp.csv',
                        '../data/icecube_10year_ps/events/IC79_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_I_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_II_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_III_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_IV_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_V_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VI_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VII_exp.csv',
                    ]
            _ice_parse = {
                    0: self.ice_parser(self._ice_data[0]),
                    1: self.ice_parser(self._ice_data[1]),
                    2: self.ice_parser(self._ice_data[2]),
                    3: self.ice_parser(self._ice_data[3]),
                    4: self.ice_parser(self._ice_data[4]),
                    5: self.ice_parser(self._ice_data[4]),
                    6: self.ice_parser(self._ice_data[4]),
                    7: self.ice_parser(self._ice_data[4]),
                    8: self.ice_parser(self._ice_data[4]),
                    9: self.ice_parser(self._ice_data[4]),
                }
            self._ice_dic = self.data_filter(_ice_parse, [2,8], [0,90], range(0,10))
            self._bkgrd_ice_data = []
            for i in self._ice_dic.keys():
                tmp_hist_data, _ = np.histogram(self._ice_dic[i][:, 1],
                                                bins=np.log10(
                                                    self._shower.e_grid))
                self._bkgrd_ice_data.append(tmp_hist_data)
            try:

                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for Atmospheric and Astro Fluxes")
                self._bkgrd = pickle.load(open(
                    "../data/background_ice.pkl", "rb"))

            except FileNotFoundError:
                _log.info("Failed to load pre-calculated tables")
                _log.info("Calculating tables for background")
                self._bkgrd = {}
                for i in config['atmospheric showers']['particles of' +
                                                       ' interest']:
                    self._bkgrd[i] = []
                for y in self._year:
                    bkd = self._detector.sim2dec(
                        self._shower.flux_results, y)

                    for i in config['atmospheric showers']['particles' +
                                                           ' of interest']:
                        self._bkgrd[i].append(bkd[i])
                pickle.dump(self._bkgrd,
                            open("../data/background_ice.pkl", "wb"))

        elif self.name == 'POne':
            try:
                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for Atmospheric and Astro Fluxes")
                self._bkgrd = pickle.load(
                    open("../data/background_pone.pkl", "rb"))
                # smearing for PONE if needed

            except FileNotFoundError:
                _log.info("Failed to load pre-calculated tables")
                _log.info("Calculating tables for background")
                self._bkgrd = self._detector.sim2dec(self._shower.flux_results,
                                                     boolean_sig=False,
                                                     boolean_smeared=self._smb)
                pickle.dump(self._bkgrd,
                            open("../data/background_pone.pkl", "wb"))

        # background counts for combined
        elif self.name == 'combined':
            self.days = 60. * 24
            self.minutes = 60.
            self.days = 60. * 24
            self.minutes = 60.
            self._ice_data = [
                        '../data/icecube_10year_ps/events/IC40_exp.csv',
                        '../data/icecube_10year_ps/events/IC59_exp.csv',
                        '../data/icecube_10year_ps/events/IC79_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_I_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_II_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_III_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_IV_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_V_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VI_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VII_exp.csv',
                    ]
            _ice_parse = {
                    0: self.ice_parser(self._ice_data[0]),
                    1: self.ice_parser(self._ice_data[1]),
                    2: self.ice_parser(self._ice_data[2]),
                    3: self.ice_parser(self._ice_data[3]),
                    4: self.ice_parser(self._ice_data[4]),
                    5: self.ice_parser(self._ice_data[4]),
                    6: self.ice_parser(self._ice_data[4]),
                    7: self.ice_parser(self._ice_data[4]),
                    8: self.ice_parser(self._ice_data[4]),
                    9: self.ice_parser(self._ice_data[4]),
                }
            self._ice_dic = self.data_filter(_ice_parse, [2,8], [0,90], range(0,10))
            
            try:
                _log.info("Trying to load pre-calculated tables for combined")
                _log.debug("Searching for Atmospheric and Astro Fluxes" +
                           " for combined")
                self._bkgrd = pickle.load(
                    open("../data/background_combined.pkl", "rb"))

            except FileNotFoundError:

                self._bkgrd = {}
                try:
                    _log.info("Trying to load pre-calculated tables IceCube")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    self._bkgrd_ice = pickle.load(open(
                        "../data/background_ice.pkl", "rb"))
                    _log.info("Trying to load observational IceCube Data files")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    #self._bkgrd_ice_data = pickle.load(open(
                    #    "../data/background_ice_ob.pkl", "rb"))
                except FileNotFoundError:
                    #self._bkgrd_ice_data = []
                    #for i in self._ice_dic.keys():
                    #    tmp_hist_data, _ = np.histogram(self._ice_dic[i][:, 1],
                    #                            bins=np.log10(
                    #                                self._shower.e_grid))
                    #    self._bkgrd_ice_data.append(tmp_hist_data)
                    #pickle.dump(self._bkgrd_ice_data,
                    #            open("../data/background_ice.pkl", "wb"))
                    # background for IceCube
                    _log.info("Failed to load pre-calculated tables")
                    _log.info("Calculating tables for background")
                    self._bkgrd_ice = {}
                    for i in config['atmospheric showers']['particles ' +
                                                           'of interest']:
                        self._bkgrd_ice[i] = []
                    for y in (self._year):
                        bkd = self._detector.sim2dec_ice(
                            self._shower.flux_results_ice, y)

                        for i in config['atmospheric showers']['particles of' +
                                                               ' interest']:
                            self._bkgrd_ice[i].append(bkd[i])
                    pickle.dump(self._bkgrd_ice,
                                open("../data/background_ice.pkl", "wb"))

                # background for P-ONE
                try:
                    _log.info("Trying to load pre-calculated tables P-ONE")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    self._bkgrd_PONE = pickle.load(
                        open("../data/background_pone.pkl", "rb"))
                except FileNotFoundError:
                    _log.info("Failed to load pre-calculated tables")
                    _log.info("Calculating tables for background")
                    self._bkgrd_PONE = self._detector.sim2dec_pone(
                                        self._shower.flux_results_pone,
                                        boolean_sig=False,
                                        boolean_smeared=self._smb)
                    pickle.dump(self._bkgrd_PONE,
                                open("../data/background_pone.pkl", "wb"))

                # combinning the background for further analysis
                for i in config['atmospheric showers']['particles ' +
                                                       'of interest']:
                    self._bkgrd_ice[i] = np.sum(self._bkgrd_ice[i], axis=0)
                    print(len(self._bkgrd_ice[i]))
                    print(len(self._bkgrd_PONE[i]))
                    self._bkgrd[i] = np.add(self._bkgrd_ice[i],
                                            self._bkgrd_PONE[i])
                pickle.dump(self._bkgrd, open(
                    '../data/background_combined.pkl', 'wb'))

    # TODO: bkgrd() -> returns

    @property
    def bkgrd(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        return self._bkgrd

    @property
    def ice_data(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        return self._bkgrd_ice_data

    def ice_parser(self, filename):

        store = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader):
                if row_num == 0:
                    continue
                store.append(row[0].split())
        store = np.array(store, dtype=float)

        return store

    def data_filter(self, event_dic, energy_range, angle_range, years):
        # filters the data in energy and angle
        filtered_dic = {}
        for year in years:
            # where method is faster as basic slicing
            energy_filter_1 = event_dic[year][np.where(event_dic[year][:, 1] < energy_range[1])]
            energy_filter_2 = energy_filter_1[np.where(energy_filter_1[:, 1] > energy_range[0])]
            high_angle = angle_range[1]
            angle_filter_1 = energy_filter_2[np.where(energy_filter_2[:, 4] < high_angle)]
            low_angle = angle_range[0]
            angle_filter_2 = angle_filter_1[np.where(angle_filter_1[:, 4] > low_angle)]
            filtered_dic[year] = angle_filter_2
        return filtered_dic

