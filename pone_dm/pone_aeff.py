# -*- coding: utf-8 -*-
# Name: pone_aeff.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Loads and processes the effective area data for P-ONE

# Imports
import logging
import numpy as np
from scipy.interpolate import UnivariateSpline
from config import config
from constants import pdm_constants
from IceCube_extraction import Icecube_data
_log = logging.getLogger(__name__)


class Aeff(object):
    """ Loads and processes the effective areas for P-ONE

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        self._const = pdm_constants()
        _log.info('Loading the effective area data')

        if config["general"]["detector"] == "POne":

            location = config['pone']['aeff location']
            _log.debug('Fetching them from ' + location)

            try:
                A_55 = np.loadtxt(location + "A_55.csv", delimiter=",")
                A_15 = np.loadtxt(location + "A_15.csv", delimiter=",")
                A_51 = np.loadtxt(location + "A_51.csv", delimiter=",")

            except FileNotFoundError:
                FileNotFoundError('Could not find the effective areas!' +
                                  ' Check the location')
            A_55 = A_55[A_55[:, 0].argsort()]
            A_15 = A_15[A_15[:, 0].argsort()]
            A_51 = A_51[A_51[:, 0].argsort()]
            A_55 = np.concatenate((np.array([[100, 0]]), A_55), axis=0)
            A_15 = np.concatenate((np.array([[100, 0]]), A_15), axis=0)
            A_51 = np.concatenate((np.array([[100, 0]]), A_51), axis=0)

            self._A_15 = UnivariateSpline(A_15[:, 0], A_15[:, 1] *
                                          self._const.msq2cmsq, k=1, s=0, ext=3)
            self._A_51 = UnivariateSpline(A_51[:, 0], A_51[:, 1] *
                                          self._const.msq2cmsq, k=1, s=0, ext=3)
            self._A_55 = UnivariateSpline(A_55[:, 0], A_55[:, 1] *
                                          self._const.msq2cmsq, k=1, s=0, ext=3)
        if config["general"]["detector"] == "IceCube":
            print("Loading Effective Area")
            _log.info("Loading Effective Area for IceCube...")


    @property
    def spl_A15(self):
        """ Fetches the spline for the effective are A15

        Parameters
        ----------
        None

        Returns
        -------
        UnivariateSpline
            The spline of the effective area
        """
        return self._A_15

    @property
    def spl_A51(self):
        """ Fetches the spline for the effective are A51

        Parameters
        ----------
        None

        Returns
        -------
        UnivariateSpline
            The spline of the effective area
        """
        return self._A_51

    @property
    def spl_A55(self):
        """ Fetches the spline for the effective are A55

        Parameters
        ----------
        None

        Returns
        -------
        UnivariateSpline
            The spline of the effective area
        """
        return self._A_55
