# -*- coding: utf-8 -*-
# Name: pdm.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# The main interface to most required functionalities

# Imports
# Native modules
from time import time
import logging
from limit_calc import Limits
import sys
import numpy as np
import yaml
# -----------------------------------------
# Package modules
from config import config
from atm_shower import Atm_Shower
from dm2nu import DM2Nu
from pone_aeff import Aeff
from bkgrd_calc import Background
from detectors import Detector
from signal_calc import Signal
import pickle
# from .limit_calc import Limits

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("pone_dm")


class PDM(object):
    """
    class: PDM
    Interace to the pdm package. This class
    stores all methods required to run the simulation
    of P-ONE's sensitivity to decaying DM
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation

    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        # --------------------------------------------------------------
        # Fetching the user inputs
        # Inputs
        self._start = time()
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)
        # --------------------------------------------------------------
        # Constructing random state for reproduciability
        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            rstate = np.random.RandomState()
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
        config["runtime"] = {"random state": rstate}
        # --------------------------------------------------------------
        # Constructing the loggers
        # Logger
        # creating file handler with debug messages
        pdm_log = logging.FileHandler(
            config["general"]["log file handler"], mode="w"
        )
        pdm_log.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])

        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        pdm_log.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        _log.addHandler(pdm_log)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        # --------------------------------------------------------------
        # Introduction
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to PDM!')
        _log.info('This package will help you model P-ONE DM sensitivities')
        # --------------------------------------------------------------
        # The atmospheric background showers
        self._shower_sim = Atm_Shower()
        _log.info('Finished the air shower simulation')
        # --------------------------------------------------------------
        # Initializing the DM methods
        self._dm_nu = DM2Nu()
        _log.info('Finished setting up the DM functions')
        # --------------------------------------------------------------
        # Fetching the effective areas -> convert to detector
        self._aeff = Aeff()
        _log.info('Finished loading the effective ares')
        # --------------------------------------------------------------
        # Constructs detector
        self._Detector = Detector(self._aeff)
        # --------------------------------------------------------------
        # Constructs signal
        self._signal = Signal(self._aeff, self._dm_nu, self._Detector)
        # --------------------------------------------------------------
        # Constructs background
        self._bkgrd = Background(self._shower_sim, self._Detector)
        # --------------------------------------------------------------
        # Setting up the limit calculations
        self._limit_calc = Limits(self._signal, self._shower_sim, self._bkgrd)
        _log.info('Finished loading the limit object')
        # -------------------------------------------------
        # Limit Calculation
        self.mass_grid = config["simulation parameters"]["mass grid"]
        self.sv_grid = config["simulation parameters"]["sv grid"]
        if config["general"]["detector"] == "IceCube":
            self._results, self._signal_data, self.prob_mat = self._limit_calc.limits(
             self.mass_grid, self.sv_grid)
            pickle.dump(self.prob_mat, open("../data/Probability_matrix_ice.pkl" , "wb"))
        else:
            self._results, self._signal_data = self._limit_calc.limits(
             self.mass_grid, self.sv_grid)
        # --------------------------------------------------------------
        # Dumping the config settings for later debugging
        _log.debug(
            "Dumping run settings into %s",
            config["general"]["config location"],
        )
        self._end = time()
        _log.info('Time of simulation %.f' % (self._end-self._start))
        with open(config["general"]["config location"], "w") as f:
            yaml.dump(config, f)
        _log.debug("Finished dump")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info("Have a great day and until next time!")
        _log.info('          /"*._         _')
        _log.info("      .-*'`    `*-.._.-'/")
        _log.info('    < * ))     ,       ( ')
        _log.info('     `*-._`._(__.--*"`./ ')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        

    @property
    def results(self):
        """ Fetches the results from the limit calculation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self._results

    @property
    def signal(self):
        """Fetches the signal counts as a 3D array with
            mass_grid x sv_grid x energy_grid
        """
        return self._signal_data

