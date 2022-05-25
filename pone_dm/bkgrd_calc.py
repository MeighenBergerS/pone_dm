# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
# from msilib.schema import File
from config import config
# from tqdm import tqdm
from atm_shower import Atm_Shower
from detectors import Detector
import pickle
import csv
import numpy as np
from scipy.interpolate import UnivariateSpline
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
            _log.info("Trying to load pre-calculated tables")
            _log.debug("Searching for Atmospheric and Astro Fluxes")
            _log.info("Failed to load pre-calculated tables")
            _log.info(
                "Calculating tables for background IceCube Projections")
            try:
                self._bkgrd = pickle.load(open(
                    "../data/background_ice.pkl", "rb"))
            except FileNotFoundError:
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
            try:
                self._bkgrd_ice_total = pickle.load(open(
                    '../data/tmp_files/background_ice_data_total.pkl',
                    'rb'
                    ))
            except FileNotFoundError:
                _log.info("Calculating tables for background IceCube Data")
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
                        5: self.ice_parser(self._ice_data[5]),
                        6: self.ice_parser(self._ice_data[6]),
                        7: self.ice_parser(self._ice_data[7]),
                        8: self.ice_parser(self._ice_data[8]),
                        9: self.ice_parser(self._ice_data[9]),
                    }
                self._ice_dic = self.data_filter(
                    _ice_parse, [2, 8], [0, 90], range(0, 10))
                self._bkgrd_ice_data = []
                for i in self._ice_dic.keys():
                    tmp_hist_data, _ = np.histogram(self._ice_dic[i][:, 1],
                                                    bins=np.log10(
                                                        self._shower.egrid))
                    self._bkgrd_ice_data.append(tmp_hist_data)
                self._bkgrd_ice_total = np.sum(self._bkgrd_ice_data, axis=0)
                pickle.dump(self._bkgrd_ice_total, open(
                    '../data/tmp_files/background_ice_data_total.pkl',
                    'wb'
                ))

        elif self.name == 'POne':
            print('pone background')
            if config['general']['pone type'] == 'old':
                try:
                    _log.info("Trying to load pre-calculated tables")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    self._bkgrd = pickle.load(
                        open("../data/background_pone.pkl", "rb"))
                    # smearing for PONE if needed

                except FileNotFoundError:
                    _log.info("Failed to load pre-calculated tables")
                    _log.info("Calculating tables for background")
                    print('Starting Calculation')
                    print(self._smb)
                    self._bkgrd = (
                         self._detector.sim2dec(self._shower.flux_results,
                                                boolean_sig=False,
                                                boolean_smeared=self._smb)
                    )
                    print('Finished Calculating Background')
                    pickle.dump(self._bkgrd,
                                open("../data/tmp_files/background_pone.pkl",
                                     "wb"))
            elif config['general']['pone type'] == 'new':
                _spacing = config['pone_christian']['spacing']
                print('Christians Background calculation started')
                self._bkgrd = (
                    self._detector.sim2dec(self._shower.flux_results,
                                           boolean_smeared=self._smb)
                )
                print('Calculation finished')
                pickle.dump(self._bkgrd, open(
                 '../data/tmp_files/Christian_/Back_christ_%f.pkl' % (_spacing),
                 'wb'))

                # _log.info("Trying to load pre-calculated tables")
                # _log.debug("Searching for Atmospheric and Astro Fluxes")
        # background counts for combined
        elif self.name == 'combined':
            self.days = 60. * 24
            self.minutes = 60.
            self.days = 60. * 24
            self.minutes = 60.

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
                    self._bkgrd_ice_total = pickle.load(open(
                        '../data/tmp_files/background_ice_data_total.pkl',
                        'rb'
                        ))
                    _log.info(
                        "Trying to load observational IceCube Data files")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    # self._bkgrd_ice_data = pickle.load(open(
                    #    "../data/background_ice_ob.pkl", "rb"))
                except FileNotFoundError:
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
                            5: self.ice_parser(self._ice_data[5]),
                            6: self.ice_parser(self._ice_data[6]),
                            7: self.ice_parser(self._ice_data[7]),
                            8: self.ice_parser(self._ice_data[8]),
                            9: self.ice_parser(self._ice_data[9]),
                        }
                    self._ice_dic = self.data_filter(
                        _ice_parse, [2, 8], [0, 90], range(0, 10))
                    self._bkgrd_ice_data = []
                    for i in self._ice_dic.keys():
                        tmp_hist_data, _ = np.histogram(self._ice_dic[i][:, 1],
                                                        bins=np.log10(
                                                            self._shower.egrid)
                                                        )
                        self._bkgrd_ice_data.append(tmp_hist_data)
                    self._bkgrd_ice_total = np.sum(self._bkgrd_ice_data,
                                                   axis=0)
                    pickle.dump(self._bkgrd_ice_total, open(
                        '../data/tmp_files/background_ice_data_total.pkl',
                        'wb'
                    ))
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
                                open("../data/tmp_files/background_pone.pkl",
                                     "wb"))

                # combinning the background for further analysis
                for i in config['atmospheric showers']['particles ' +
                                                       'of interest']:
                    self._bkgrd_ice[i] = np.sum(self._bkgrd_ice[i], axis=0)
                    print(len(self._bkgrd_ice[i]))
                    print(len(self._bkgrd_PONE[i]))
                    self._bkgrd[i] = np.add(self._bkgrd_ice[i],
                                            self._bkgrd_PONE[i])
                pickle.dump(self._bkgrd, open(
                    '../data/tmp_files/background_combined.pkl', 'wb'))

    # TODO: bkgrd() -> returns

    def odr_fit_phi(self, A, x):
        spl_phi = UnivariateSpline(self._shower._egrid,
                                   np.sum(np.array(self._bkgrd['numu']),
                                          axis=0),
                                   k=1, s=0)
        return (A[0] * spl_phi(x + A[1]) + A[2])

    @property
    def bkgrd_data(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        # e_bin = self.width2grid(self._shower._egrid)
        # self._bkgrd_ob_total_sp = (
        #     UnivariateSpline(e_bin,
        #                      np.array(self._bkgrd_ice_total),
        #                      k=1, s=0)(self._shower._egrid)
        #     )
        return self._bkgrd_ice_total  # self._bkgrd_ob_total_sp

    @property
    def bkgrd(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        return self._bkgrd

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
            energy_filter_1 = (
             event_dic[year][np.where(event_dic[year][:, 1] < energy_range[1])]
            )
            energy_filter_2 = (
             energy_filter_1[np.where(energy_filter_1[:, 1] > energy_range[0])]
            )
            high_angle = angle_range[1]
            angle_filter_1 = (
             energy_filter_2[np.where(energy_filter_2[:, 4] < high_angle)]
            )
            low_angle = angle_range[0]
            angle_filter_2 = (
             angle_filter_1[np.where(angle_filter_1[:, 4] > low_angle)]
            )
            filtered_dic[year] = angle_filter_2
        return filtered_dic

    def width2grid(self, a: np.array):
        m_a = []
        for i, e in enumerate(a):
            if i == 0:
                m_a.append(a[i])
            elif i == len(a)-1:
                break
            else:
                m_a.append((a[i] + a[i+1]) / 2)
        return m_a
