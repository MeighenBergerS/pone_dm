# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
from config import config
from atm_shower import Atm_Shower
from detectors import Detector
import pickle
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

        # Check this again
        if self.name == "IceCube":
            self.days = 60. * 24
            self.minutes = 60.

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
                    open("../data/background_pone_unsm.pkl", "rb"))
                # smearing for PONE if needed

            except FileNotFoundError:
                _log.info("Failed to load pre-calculated tables")
                _log.info("Calculating tables for background")
                self._bkgrd = self._detector.sim2dec(self._shower.flux_results,
                                                     boolean_sig=False)
                pickle.dump(self._bkgrd,
                            open("../data/background_pone_unsm.pkl", "wb"))

        # background counts for combined
        elif self.name == 'combined':
            self.days = 60. * 24
            self.minutes = 60.
            try:
                _log.info("Trying to load pre-calculated tables for combined")
                _log.debug("Searching for Atmospheric and Astro Fluxes" +
                           " for combined")
                self._bkgrd = pickle.load(
                    open("../data/background_combined.pkl", "rb"))

            except FileNotFoundError:
                # background for IceCube

                self._bkgrd = {}
                try:
                    _log.info("Trying to load pre-calculated tables IceCube")
                    _log.debug("Searching for Atmospheric and Astro Fluxes")
                    self._bkgrd_ice = pickle.load(open(
                        "../data/background_ice.pkl", "rb"))
                except FileNotFoundError:
                    _log.info("Failed to load pre-calculated tables")
                    _log.info("Calculating tables for background")
                    self._bkgrd_ice = {}
                    for i in config['atmospheric showers']['particles ' +
                                                           'of interest']:
                        self._bkgrd_ice[i] = []
                    for y in self._year:
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
                                        boolean_combined=True)
                    pickle.dump(self._bkgrd_PONE,
                                open("../data/background_pone.pkl", "wb"))

                # combinning the background for further analysis
                for i in self._bkgrd.keys():
                    for y in self._bkgrd_ice[i].keys():
                        self._bkgrd_ice[i] = np.sum(self._bkgrd_ice[i], axis=0)
                    self._bkgrd[i] = self._bkgrd_ice[i] + self._bkgrd_PONE[i]
                pickle.dump(self._bkgrd, open('../data/' +
                                              'background_combined.pkl', 'wb'))

    # TODO: bkgrd() -> returns

    @property
    def bkgrd(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        return self._bkgrd
