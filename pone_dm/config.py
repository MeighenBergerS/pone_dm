# -*- coding: utf-8 -*-
# Name: config.py
# Authors: Stephan Meighen-Berger
# Config file for the pdm package

import logging
import numpy as np
from typing import Dict, Any
import yaml

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # Random state seed
        "random state seed": 1337,
        # Output level
        'debug level': logging.ERROR,
        # Location of logging file handler
        "log file handler": "../run/pdm.log",
        # Dump experiment config to this location
        "config location": "../run/config.txt",
    },
    "simulation parameters": {
        "mass grid": np.logspace(3, 6, 5),
        "snu grid": np.logspace(-26, -23, 5),
        "uptime": 5 * 365 * 24 * 60 * 60,
        "low enery cutoff": 5e2  # GeV
    },
    ###########################################################################
    # Atmospheric showers input
    ###########################################################################
    "atmospheric showers": {
        # native mceq or built one
        'native mceq': False,
        # Path to the built version
        'path to mceq': 'C:\\Users\\steph\\Documents\\MCEq\\',
        # The atmosphere
        'atmosphere' : ('CORSIKA', ("Karlsruhe", None)),
        # The interaction model
        'interaction model' : 'SIBYLL2.3c',
        # Primary model
        'primary model' : 'H4a',
        # Angles of interest currently not custom
        'theta angles' : np.array([0., 5., 10., 20., 30., 45., 60., 70., 90.]),
        # Particles of interest
        'particles of interest' : ['numu', 'nue', 'nutau']
    },
    ###########################################################################
    # P-ONE
    ###########################################################################
    "pone": {
        'aeff location' : '..\\data\\'
    }
}


class ConfigClass(dict):
    """ The configuration class. This is used
    by the package for all parameter settings. If something goes wrong
    its usually here.

    Parameters
    ----------
    config : dic
        The config dictionary

    Returns
    -------
    None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: Update this
    def from_yaml(self, yaml_file: str) -> None:
        """ Update config with yaml file

        Parameters
        ----------
        yaml_file : str
            path to yaml file

        Returns
        -------
        None
        """
        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        self.update(yaml_config)

    # TODO: Update this
    def from_dict(self, user_dict: Dict[Any, Any]) -> None:
        """ Creates a config from dictionary

        Parameters
        ----------
        user_dict : dic
            The user dictionary

        Returns
        -------
        None
        """
        self.update(user_dict)


config = ConfigClass(_baseconfig)
