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
        self.bkgrd_down = {}
        
        self.bkgrd_horizon = {}
        
        # backgorund dictionary repositioned
        self._bkgrd={}
        
        for i in (config['atmospheric showers']['particles of interest']):
            self.bkgrd_down[i] = []
            down_angles = []
            self.bkgrd_horizon[i] = []
            horizon_angles = []
        # Astrophysical is the same everywhere
            for angle in config['atmospheric showers']['theta angles']:
                rad = np.deg2rad(np.abs(angle - 90.))
                
                # Downgoing
                if np.pi / 3 <= rad <= np.pi / 2:
                    
                    down_angles.append(rad)
                    self.bkgrd_down[i].append(
                        (self._shower.flux_results[angle][i] +
                         self._dphi_astro(self._egrid)) * self._uptime *
                        self._ewidth * self._aeff.spl_A15(self._egrid)
                    )
                    
                # Horizon
                else:
                    horizon_angles.append(rad)
                    self.bkgrd_horizon[i].append(
                        (self._shower.flux_results[angle][i] +
                         self._dphi_astro(self._egrid)) * self._uptime *
                        self._ewidth * self._aeff.spl_A55(self._egrid)
                    )
                    
            # Converting to numpy arrays
            self.bkgrd_down[i] = np.array(self.bkgrd_down[i])
            self.bkgrd_horizon[i] = np.array(self.bkgrd_horizon[i])
            down_angles = np.array(down_angles)
            horizon_angles = np.array(horizon_angles)
            
            # Integrating
                        
            self._bkgrd[i] = np.zeros_like(self._egrid)
            sorted_ids = np.argsort(down_angles)
            
            # Downgoing
            self._bkgrd[i] += np.trapz(self.bkgrd_down[i][sorted_ids],
                                         x=down_angles[sorted_ids], axis=0)
            
            # Horizon we assume it is mirrored
            sorted_ids = np.argsort(horizon_angles)
            self._bkgrd[i] += 2. * np.trapz(self.bkgrd_horizon[i][sorted_ids],
                                              x=horizon_angles[sorted_ids], axis=0)
            
            # Upgoing we assume the same flux for all
            self._bkgrd[i] += (
                (np.pi / 2 - np.pi / 3) *
                (self._shower.flux_results[0.][i] +
                 self._dphi_astro(self._egrid)) * self._uptime *
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
        
        # Storage for the new signal fluxes
        self._signal_flux = {}
        self._signal_counts = {}
        
        y = {}
        
        #for more generations adding the loop ----
        for i in (config['atmospheric showers']['particles of interest']):
            
            for m in mass_grid:
                self._signal_flux[m] = {}
                self._signal_counts[m] = {}
                
            _log.info('Starting the limit calculation')
            
            # The low energy cut off
            self._t_d = self._find_nearest(self._egrid, 5e2)
            self._limit_scan_grid_base = np.array([[
                (self._signal_calc(self._egrid, mass,
                    sv, config['atmospheric showers']['theta angles']
                )**2.)[self._t_d:] /
                self._bkgrd[i][self._t_d:]
                for mass in mass_grid] for sv in tqdm(sv_grid)])
            y[i] = np.array([[
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
        extra = self._dmnu.extra_galactic_flux(egrid, mass, sv)
        self._extra_flux = extra
        
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
        
        self._signal_flux[mass][sv] = self._extra_flux
        
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
        total_new_counts = (
            (self._extra + self._ours) /
            config["advanced"]["scaling correction"]  # Some unknown error
        )
        self._signal_counts[mass][sv] = total_new_counts
        return total_new_counts

    
    
    def _find_nearest(self, array, value):
        """ Add description
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _dphi_astro(self, E):
        """
        Astrophysical flux because of muon background as per the power
        law described in https://arxiv.org/pdf/1907.11266.pdf

        Add description
        """
        return 1e-18*1.66* ((E/1e5)**(-2.53))     # 1e-18 * 6.45 * (E / 1e5)**(-2.89)
