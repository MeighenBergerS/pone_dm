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
        "detector": ["IceCube",   "POne", 'combined'],
        "pone type": ["new", "old"],
        "year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'density': ['NFW', 'Burkert'],
        "channel": ["W", "\[Tau]", "b", "All"],
    },
    "simulation parameters": {
        "mass grid": np.logspace(2, 6, 10),
        "sv grid": np.logspace(-26, -21, 10),
        "uptime": 10 * 365 * 24 * 60 * 60,
        "low energy cutoff": 1.0e3,  # GeV
        "high energy cutoff": 1e6,  # GeV
        "DM type k": 2
    },
    ###########################################################################
    # Atmospheric showers input
    ###########################################################################
    "atmospheric showers": {
        # native mceq or built one
        'native mceq': True,
        # Path to the built version
        'path to mceq': '/home/kruteesh/miniconda3/envs/pdm/lib/python3.9' +
        '/site-packages/MCEq/',
        # The atmosphere
        'atmosphere': ('CORSIKA', ("Karlsruhe", None)),
        # The interaction model
        'interaction model': 'SIBYLL2.3c',
        # Primary model
        'primary model': 'H4a',
        # Angles of interest currently not custom
        'theta angles': range(0, 91, 1),
        # Particles of interest
        'particles of interest': ['numu', "nue", "nutau"]
    },
    ###########################################################################
    # P-ONE
    ###########################################################################
    "pone": {
            'aeff location': '../data/',
            "specific particle scaling": {
                "numu": 1.,
                "nue": 1.,
                "nutau": 1.},
            'smearing': ['smeared', 'unsmeared'],
            'low E sigma': {
                'numu': [0.45, 0.35, 0.25, 0.15],
                'nue': [0.55, 0.45, 0.35, 0.25],
                'nutau': [0.55, 0.45, 0.35, 0.25]},
            'high E sigma': {
                'numu': [0.25, 0.15, 0.10, 0.09],
                'nue': [0.15, 0.12, 0.09, 0.07],
                'nutau': [0.15, 0.12, 0.09, 0.07]
                }
            # Entries: numu, nue, nutau
    },
    'pone_christian': {
        'aeff location': '../data/',
        'angles': [0, 1.46928358,  4.41172579,  7.36588583, 10.33989089,
                   13.3423638, 16.38266985, 19.47122063, 22.61986495,
                   25.84241287, 29.15536543, 32.57897039, 36.13881466,
                   39.86834155, 43.81306146, 48.03811117,
                   52.64314803, 57.7957725, 63.82304783, 71.5713304, 90],
        'spacing': [],
        'hit': [],
        'module': [],
        'pos res': [],


    },
    ###########################################################################
    # Advanced
    ###########################################################################
    "advanced": {
        "integration grid lopez": np.logspace(-3, 12, 151),
        "construction grid _d": np.logspace(-13, 16, 600),
        "_d storage": "../data/",
        "atmospheric storage": "../data/",
        "scaling correction": 1e-3
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
