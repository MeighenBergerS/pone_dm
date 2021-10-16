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
_log = logging.getLogger(__name__)


class Background(object):
    """
    Class to calculate background counts for both P-One and IceCube detectors
    """
    def __init__(self, shower_sim: Atm_Shower, detector: Detector):

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
                self._bkgrd = self._detector.sim2dec(
                    self._shower.flux_results, config['general']['year'])
                pickle.dump(self._bkgrd,
                            open("../data/background_ice.pkl", "wb"))

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
