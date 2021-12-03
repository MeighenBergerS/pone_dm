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
_log = logging.getLogger("pone_dm")


class Background(object):
    """
    Class to calculate background counts for both P-One and IceCube detectors
    """
    def __init__(self, shower_sim: Atm_Shower, detector: Detector, year=config['general']['year']):
        
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
                self._bkgrd = []

                for y in self._year:
                    bkd, _ = self._detector.sim2dec(
                        self._shower.flux_results, y)
                    self._bkgrd.append(bkd)
                
                pickle.dump(self._bkgrd,
                            open("../data/background_ice.pkl", "wb"))
                pickle.dump(self.eff_are_dic, open("../data/eff_area_ice.pkl",
                                                   "wb"))


        elif self.name == 'POne':

            try:
                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for Atmospheric and Astro Fluxes")
                self._bkgrd = pickle.load(
                    open("../data/background_pone.pkl", "rb"))

            except FileNotFoundError:
                _log.info("Failed to load pre-calculated tables")
                _log.info("Calculating tables for background")
                self._bkgrd = self._detector.sim2dec(self._shower.flux_results)
                pickle.dump(self._bkgrd,
                            open("../data/background_pone.pkl", "wb"))

    # TODO: bkgrd() -> returns

    @property
    def bkgrd(self):
        """Returns backgorund counts dict [ label : Neutrino Flavour ]
        """
        return self._bkgrd
