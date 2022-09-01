# -*- coding: utf-8 -*-
# Name: signal_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the signal counts for DM -> Neutrino

# Imports
import logging
import numpy as np
from config import config
from pone_aeff import Aeff
from dm2nu import DM2Nu
from constants import pdm_constants
from detectors import Detector
_log = logging.getLogger("pone_dm")


class Signal(object):
    """ Class of methods to calculate signal at detector from DM -> Neutrinos

    Parameters
    ----------
    aeff: Aeff
        An Aeff object
    dm_nu: DM2Nu
        A DM2Nu object
    detector: Detector
        A Detector object
    """
    def __init__(self, aeff: Aeff, dmnu: DM2Nu,
                 detector: Detector, year=config['general']['year']):
        self._aeff = aeff

        self._dmnu = dmnu
        self._detector = detector
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']
        self._year = year
        self._ewidth = self._aeff._ewidth
        self._egrid = self._aeff._egrid
        self.name = config['general']['detector']

        self._s_pone = self._signal_calc_pone
        self._s_ice = self._signal_calc_ice
        self._pone_smearing = config['pone']['smearing']
        self._channel = config['general']["channel"]
        if self._pone_smearing == 'smeared':
            self._bool_smea = True
        elif self._pone_smearing == 'unsmeared':
            self._bool_smea = False
        if self.name == 'IceCube':
            print(self.name)
            self._signal_calc = self._signal_calc_ice
        elif self.name == 'POne':
            if config['general']['pone type'] == 'old':
                print(self.name)
                self._signal_calc = self._signal_calc_pone
            elif config['general']['pone type'] == 'new':
                print('Christians Effective Areas are being used')
                self._hit = config['pone_christian']['hit']
                self._module = config['pone_christian']['module']
                self._spacing = config['pone_christian']['spacing']
                self._pos_res = config['pone_christian']['pos res']
                self._signal_calc = self._signal_calc_pone_christ

        elif self.name == 'combined':
            print(self.name)
            self._signal_calc = self._signal_calc_combined

        if self._channel != "All":
            self._extra_dm = self._dmnu.extra_galactic_flux_c
            self._galac_dm = self._dmnu.galactic_flux_c

    @property
    def signal_calc(self):
        """Returns appropriate signal calculation function
        total_counts : np.array
        """
        return self._signal_calc

    @property
    def signal_calc_pone(self):
        """For combined analysis we need the seperate function
        for P-ONE
        """
        return self._s_pone

    @property
    def signal_calc_ice(self):
        """For combined analysis we need the seperate function
        for IceCube
        """
        return self._s_ice

    def _signal_calc_ice(self, egrid: np.array, mass: float,
                         sv: float):
        """ Calculates the expected signal given the mass, sigma*v and angle

        Parameters
        ----------
        egrid : np.array
            The energy grid to calculate on
        mass : float
            The DM mass
        sv : float
            The sigma * v of interest
        angle_grid : np.array
            The incoming angles

        Returns
        -------
        total_new_counts : np.array (len(year),len(E_grid))
            The total new counts
        """
        # Extra galactic
        _extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)

        # Galactic
        total_new_counts = []
        # TODO: Need to configure for IceCube ------

        _ours = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d + self._const.J_p + self._const.J_s
        )
        # Converting fluxes into counts with effective area of IceCube !!!!
        #  These steps take a lot of time !!!!
        total_flux = _ours+_extra
        if self.name == 'combined':
            for y in self._year:
                _log.info("combined signal ice year =" +
                          "%e, mass = %.1e, sv = %.1e" % (y, mass, sv))
                total_new_counts.append(
                    np.array(self._detector.sim2dec_ice(total_flux,
                                                        y)['numu']))
        elif self.name == 'IceCube':
            for y in self._year:
                _log.info(" signal ice year =" +
                          "%e, mass = %.1e, sv = %.1e" % (y, mass, sv))
                total_new_counts.append(
                    np.array(self._detector.sim2dec(total_flux,
                                                    y)['numu']))

        # the sim_to_dec omits the dict but we assume
        # same for all neutrino flavours

        return total_new_counts

    def _signal_calc_pone_christ(self, egrid: np.array, mass: float,
                                 sv: float):
        """Calculates the expected signal given the mass, sigma*v
        with christian's effective area file
        total_new_counts : 
        """
        _extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)

        # Galactic
        total_new_counts = []
        # TODO: Need to configure for IceCube ------

        _ours = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d + self._const.J_p + self._const.J_s
        )
        # Converting fluxes into counts with effective area of IceCube !!!!
        #  These steps take a lot of time !!!!
        total_flux = _ours+_extra
        for y in self._year:
            _log.info(" signal ice year =" +
                      "%e, mass = %.1e, sv = %.1e" % (y, mass, sv))
            tmp_y_counts, tmp_counts_y_ang = (
                self._detector.sim2dec(total_flux,
                                       boolean_sig=True,
                                       boolean_smeared=self._bool_smea))

            total_new_counts.append(np.array(tmp_y_counts['numu']))

        # the sim_to_dec omits the dict but we assume
        return total_new_counts

    def _signal_calc_pone(self, egrid: np.array, mass: float,
                          sv: float):
        """ Calculates the expected signal given the mass, sigma*v and angle

        Parameters
        ----------
        egrid : np.array
            The energy grid to calculate on
        mass : float
            The DM mass
        sv : float
            The sigma * v of interest
        angle_grid : np.array
            The incoming angles

        Returns
        -------
        total_new_counts : dictionary ( #numu , # nue, #nutau )
        """

        # Extra galactic

        extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)
        _flux = {}
        _flux[15] = {}
        _flux[85] = {}
        _flux[120] = {}
        # Galactic
        for i in config['atmospheric showers']['particles of interest']:
            _flux[15][i] = np.array(extra) + self._dmnu.galactic_flux(
                egrid, mass, sv,
                config['simulation parameters']["DM type k"],
                self._const.J_d1 + self._const.J_p1 + self._const.J_s1
            )

            _flux[85][i] = np.array(extra) + self._dmnu.galactic_flux(
                egrid, mass, sv,
                config['simulation parameters']["DM type k"],
                self._const.J_d2 + self._const.J_p2 + self._const.J_s2
            )

            _flux[120][i] = np.array(extra) + self._dmnu.galactic_flux(
                egrid, mass, sv,
                config['simulation parameters']["DM type k"],
                self._const.J_d3 + self._const.J_p3 + self._const.J_s3
            )

        if self.name == 'combined':
            total_counts = (
                self._detector.sim2dec_pone(_flux,
                                            boolean_sig=True,
                                            boolean_smeared=self._bool_smea))
        else:
            total_counts = (
                self._detector.sim2dec(_flux, boolean_sig=True,
                                       boolean_smeared=self._bool_smea))
            # smearing for PONE if needed

        return total_counts

    def _signal_calc_combined(self, egrid, mass, sv):
        signal_ice = np.sum(self._signal_calc_ice(egrid, mass, sv), axis=0)
        signal_pone = self._signal_calc_pone(egrid, mass, sv)
        signal_dic = {}
        for i in config['atmospheric showers']['particles of interest']:
            signal_dic[i] = signal_ice + signal_pone[i]
        return signal_dic

    def _find_nearest(self, array: np.array, value: float):

        """ Returns: index of the nearest vlaue of an array to the given number
        --------------
        idx :  float
        """
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return idx

