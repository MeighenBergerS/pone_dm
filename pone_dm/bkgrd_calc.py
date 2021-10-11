# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
from config import config
from pone_aeff import Aeff
from dm2nu import DM2Nu
from atm_shower import Atm_Shower
from constants import pdm_constants
from detectors import Detector

_log = logging.getLogger(__name__)


class Background(object):
    """
    Class to calculate background counts for both P-One and IceCube detectors
    """
    def __init__(self, aeff: Aeff, dm_nu: DM2Nu, shower_sim: Atm_Shower,
                 detector: Detector):

        self._aeff = aeff
        self._dmnu = dm_nu
        self._shower = shower_sim
        self._detector = detector
        self._egrid = self._shower.egrid
        self._ewidth = self._shower.ewidth
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']

        _log.info('Initializing the Limits object')
        _log.info('Preliminary calculations')
        _log.debug('The total atmospheric flux')

        # Check this again
        self.bkgrd = self._detector(self._aeff,
                                    self._dmnu, self._shower).sim2dec
