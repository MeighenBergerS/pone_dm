# -*- coding: utf-8 -*-
# Name: limit_calc.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Calculates the limits

# Imports
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
from .config import config
from .pone_aeff import Aeff
from .dm2nu import DM2Nu
from .atm_shower import Atm_Shower
from .constants import pdm_constants

_log = logging.getLogger(__name__)

class Limits(object):
    """ Collection of methods to calculate the limits

    Parameters
    ----------
    aeff: Aeff
        An Aeff object
    dm_nu: DM2Nu
        A DM2Nu object
    shower_sim: Atm_Shower
        A Atm_Shower object
    """
    def __init__(self, aeff: Aeff, dm_nu: DM2Nu, shower_sim: Atm_Shower):
        _log.info('Initializing the Limits object')
        self._aeff = aeff
        self._dmnu = dm_nu
        self._shower = shower_sim
        self._egrid = self._shower.egrid
        self._ewidth = self._shower.ewidth
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']
        _log.info('Preliminary calculations')
        _log.debug('The total atmospheric flux')
        # The fluxes convolved with the effective area
        self._numu_bkgrd_down = []
        down_angles = []
        self._numu_bkgrd_horizon = []
        horizon_angles = []
        for angle in config['atmospheric showers']['theta angles']:
            rad = np.deg2rad(np.abs(angle - 90.))
            # Downgoing
            if np.pi / 3 <= rad <= np.pi / 2:
                down_angles.append(rad)
                self._numu_bkgrd_down.append(
                    self._shower.flux_results[angle]['numu'] * self._uptime *
                    self._ewidth * self._aeff.spl_A15(self._egrid)
                )
            # Horizon
            else:
                horizon_angles.append(rad)
                self._numu_bkgrd_horizon.append(
                    self._shower.flux_results[angle]['numu'] * self._uptime *
                    self._ewidth * self._aeff.spl_A55(self._egrid)
                )
        # Converting to numpy arrays
        self._numu_bkgrd_down = np.array(self._numu_bkgrd_down)
        self._numu_bkgrd_horizon = np.array(self._numu_bkgrd_horizon)
        down_angles = np.array(down_angles)
        horizon_angles = np.array(horizon_angles)
        # Integrating
        self._numu_bkgrd = np.zeros_like(self._egrid)
        sorted_ids = np.argsort(down_angles)
        # Downgoing
        self._numu_bkgrd += np.trapz(self._numu_bkgrd_down[sorted_ids],
                                     x=down_angles[sorted_ids], axis=0)
        # Horizon we assume it is mirrored
        sorted_ids = np.argsort(horizon_angles)
        self._numu_bkgrd += 2. * np.trapz(self._numu_bkgrd_horizon[sorted_ids],
                                          x=horizon_angles[sorted_ids], axis=0)
        # Upgoing we assume the same flux for all
        self._numu_bkgrd += (
            (np.pi / 2 - np.pi / 3) *
            self._shower.flux_results[0.]['numu'] * self._uptime *
            self._ewidth * self._aeff.spl_A51(self._egrid)
        )

    def limit_calc(self,
        mass_grid=config["simulation parameters"]["mass grid"],
        sv_grid=config["simulation parameters"]["sv grid"]) -> np.array:
        """ Scans the masses and sigma*nu and calculates
        the corresponding limits

        Parameters
        ----------
        mass_grid : np.array
            The masses to scan
        sv_grid : np.array
            The sigma * v grid

        Returns
        -------
        list
            The resulting chi values
        """
        _log.info('Starting the limit calculation')
        # The low energy cut off
        t_d = self._find_nearest(self._egrid, 5e2)
        self._limit_scan_grid_base = np.array([[
            (self._signal_calc(self._egrid, mass,
                sv, config['atmospheric showers']['theta angles']
            )**2.)[:t_d] / self._numu_bkgrd[:t_d]
            for mass in mass_grid] for sv in tqdm(sv_grid)])
        y = np.array([[
            chi2.sf(np.sum(np.nan_to_num(x)), 2)
            for x in k] for k in self._limit_scan_grid_base
        ])
        return y
        

    def _signal_calc(self, egrid: np.array, mass: float,
                     sv: float, angle_grid: np.array) -> np.array:
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
        total_new_counts : np.array
            The total new counts
        """
        # Extra galactic
        _log.debug('The energy grid used for the new' +
                    ' flux calculation has shape ' + str(egrid.shape))
        extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)
        self._extra_flux = extra
        _log.debug('The extra galactic component has shape ' + str(extra.shape))
        # Galactic
        ours_15 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d1 + self._const.J_p1 + self._const.J_s1
        )
        ours_55 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d2 + self._const.J_p2 + self._const.J_s2
        )
        ours_51 = self._dmnu.galactic_flux(
            egrid, mass, sv,
            config['simulation parameters']["DM type k"],
            self._const.J_d3 + self._const.J_p3 + self._const.J_s3
        )
        # Convolving
        down_angles = []
        horizon_angles = []
        extra_down = []
        extra_hor = []
        ours_down = []
        ours_hor = []
        for angle in angle_grid:
            rad = np.deg2rad(np.abs(angle - 90.))
            # Downgoing
            if np.pi / 3 <= rad <= np.pi / 2:
                down_angles.append(rad)
                extra_down.append(
                    extra * self._uptime *
                    self._ewidth * self._aeff.spl_A15(self._egrid)
                )
                ours_down.append(
                    ours_15 * self._uptime *
                    self._ewidth * self._aeff.spl_A15(self._egrid)
                )
            # Horizon
            else:
                horizon_angles.append(rad)
                extra_hor.append(
                    extra * self._uptime *
                    self._ewidth * self._aeff.spl_A55(self._egrid)
                )
                ours_hor.append(
                    ours_55 * self._uptime *
                    self._ewidth * self._aeff.spl_A55(self._egrid)
                )
        # Converting to numpy arrays
        extra_down = np.array(extra_down)
        extra_hor = np.array(extra_hor)
        ours_down = np.array(ours_down)
        ours_hor = np.array(ours_hor)
        down_angles = np.array(down_angles)
        horizon_angles = np.array(horizon_angles)
        # Integrating
        # Extra
        self._extra = np.zeros_like(self._egrid)
        # Downgoing
        sorted_ids = np.argsort(down_angles)
        self._extra += np.trapz(extra_down[sorted_ids],
                                x=down_angles[sorted_ids], axis=0)
        # Horizon we assume it is mirrored
        sorted_ids = np.argsort(horizon_angles)
        self._extra += 2. * np.trapz(extra_hor[sorted_ids],
                                     x=horizon_angles[sorted_ids], axis=0)
        # Upgoing we assume the same flux for all
        self._extra += (
            (np.pi / 2 - np.pi / 3) *
            extra * self._uptime *
            self._ewidth * self._aeff.spl_A51(self._egrid)
        )
        # Ours
        self._ours = np.zeros_like(self._egrid)
        # Downgoing
        sorted_ids = np.argsort(down_angles)
        self._ours += np.trapz(ours_down[sorted_ids],
                               x=down_angles[sorted_ids], axis=0)
        # Horizon we assume it is mirrored
        sorted_ids = np.argsort(horizon_angles)
        self._ours += 2. * np.trapz(ours_hor[sorted_ids],
                                    x=horizon_angles[sorted_ids], axis=0)
        # Upgoing we assume the same flux for all
        self._ours += (
            (np.pi / 2 - np.pi / 3) *
            ours_51 * self._uptime *
            self._ewidth * self._aeff.spl_A51(self._egrid)
        )
        total_new_counts = self._extra + self._ours
        return total_new_counts

    def _find_nearest(self, array, value):
        """ Add description
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
