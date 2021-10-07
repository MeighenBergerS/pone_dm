# -*- coding: utf-8 -*-
# Name: IceCube_extraction.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Quick and dirty interface to MCEq for the pdm package

# Imports
import logging

from numpy.core.records import array
from pone_aeff import Aeff
from dm2nu import DM2Nu
from atm_shower import Atm_Shower
from constants import pdm_constants
import pickle
from config import config
import numpy as np
from tqdm import tqdm
import csv

from scipy.interpolate import UnivariateSpline
_log = logging.getLogger(__name__)


class Icecube_data(object):
    """
    Class to extract counts depending on the type of Detectors
    IceCube and POne are only two choices there are as of now---------

    """

    def __init__(self, aeff: Aeff, dm_nu: DM2Nu, shower_sim: Atm_Shower):

        self.name = config["general"]["detector"]
        self._aeff = aeff
        self._dmnu = dm_nu
        self._shower = shower_sim
        self._egrid = self._shower.egrid
        self._ewidth = self._shower.ewidth
        self._const = pdm_constants()
        self._uptime = config['simulation parameters']['uptime']
        
        if self.name == "IceCube":
            self.year = 5  # Change it with config file----------
            try:

                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for Atmospheric and Astrophysical Fluxes")
                self.minutes = 60.
                self.surface_fluxes = pickle.load(open("../data/surf_store_v1.p", "rb"))
                self.days = 60. * 24
                self.particle_counts_smeared_unin = pickle.load(open('../data/atmos_all.p',"rb"))
                self.astro_counts_smeared_unin = pickle.load(open("../data/astro_all.p", "rb"))
                self._egrid = self.surface_fluxes[0][0]

            except FileNotFoundError:
                self.eff_areas = [
                    '../data/icecube_10year_ps/irfs/IC40_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC59_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC79_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_I_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_II_effectiveArea.csv',
                ]

                self.eff_dic = {
                        0: self.ice_parser(self.eff_areas[0]),
                        1: self.ice_parser(self.eff_areas[1]),
                        2: self.ice_parser(self.eff_areas[2]),
                        3: self.ice_parser(self.eff_areas[3]),
                        4: self.ice_parser(self.eff_areas[4]),
                        5: self.ice_parser(self.eff_areas[4]),
                        6: self.ice_parser(self.eff_areas[4]),
                        7: self.ice_parser(self.eff_areas[4]),
                        8: self.ice_parser(self.eff_areas[4]),
                        9: self.ice_parser(self.eff_areas[4]),
                    }

                # Loading smearing
                # log10(E_nu/GeV)_min, log10(E_nu/GeV)_max, Dec_nu_min[deg], Dec_nu_max[deg], log10(E/GeV), PSF_min[deg], PSF_max[deg],
                # AngErr_min[deg], AngErr_max[deg], Fractional_Counts
                self.smearing_sets = [
                        '../data/icecube_10year_ps/irfs/IC40_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC59_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC79_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_I_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                        '../data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
                    ]

                self.smearing_dic = {
                        0: self.ice_parser(self.smearing_sets[0]),
                        1: self.ice_parser(self.smearing_sets[1]),
                        2: self.ice_parser(self.smearing_sets[2]),
                        3: self.ice_parser(self.smearing_sets[3]),
                        4: self.ice_parser(self.smearing_sets[4]),
                        5: self.ice_parser(self.smearing_sets[5]),
                        6: self.ice_parser(self.smearing_sets[6]),
                        7: self.ice_parser(self.smearing_sets[7]),
                        8: self.ice_parser(self.smearing_sets[8]),
                        9: self.ice_parser(self.smearing_sets[9]),
                        }
                # MJD, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg], Azimuth[deg],Zenith[deg]
                self.data_sets = [
                        '../data/icecube_10year_ps/events/IC40_exp.csv',
                        '../data/icecube_10year_ps/events/IC59_exp.csv',
                        '../data/icecube_10year_ps/events/IC79_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_I_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_II_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_III_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_IV_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_V_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VI_exp.csv',
                        '../data/icecube_10year_ps/events/IC86_VII_exp.csv',
                    ]
                self.event_dic = {
                        0: self.ice_parser(self.data_sets[0]),
                        1: self.ice_parser(self.data_sets[1]),
                        2: self.ice_parser(self.data_sets[2]),
                        3: self.ice_parser(self.data_sets[3]),
                        4: self.ice_parser(self.data_sets[4]),
                        5: self.ice_parser(self.data_sets[5]),
                        6: self.ice_parser(self.data_sets[6]),
                        7: self.ice_parser(self.data_sets[7]),
                        8: self.ice_parser(self.data_sets[8]),
                        9: self.ice_parser(self.data_sets[9]),
                    }
                # MJD, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg], Azimuth[deg], Zenith[deg]
                self.uptime_sets = [
                        '../data/icecube_10year_ps/uptime/IC40_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC59_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC79_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_I_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_II_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_III_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_IV_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_V_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_VI_exp.csv',
                        '../data/icecube_10year_ps/uptime/IC86_VII_exp.csv',
                    ]
                self.uptime_dic = {
                        0: self.ice_parser(self.uptime_sets[0]),
                        1: self.ice_parser(self.uptime_sets[1]),
                        2: self.ice_parser(self.uptime_sets[2]),
                        3: self.ice_parser(self.uptime_sets[3]),
                        4: self.ice_parser(self.uptime_sets[4]),
                        5: self.ice_parser(self.uptime_sets[5]),
                        6: self.ice_parser(self.uptime_sets[6]),
                        7: self.ice_parser(self.uptime_sets[7]),
                        8: self.ice_parser(self.uptime_sets[8]),
                        9: self.ice_parser(self.uptime_sets[9]),
                    }
                self.uptime_tot_dic = {}
                for year in range(10):
                    self.uptime_tot_dic[year] = np.sum(np.diff(self.uptime_dic[self.year])) * self.days
                # Loading simulation results
                self.surface_fluxes = pickle.load(open("../data/surf_store_v1.p", "rb"))
                # Adding 90 deg
                self.surface_fluxes[90] = self.surface_fluxes[89]
                self._egrid = self.surface_fluxes[0][0]
            self.sim2dec = self._sim_to_dec

        if self.name == "P-ONE":
            #
            #
            #
            #
            self.sim2dec = self._simdec_Pone(self._shower.flux_results,self.year)
# ------------------------------------------
# Icecube functions ------

    def spl_e_areas(self):
        """
        for fetching effective areas
        """
        return self.spl_e_area

    def astro_flux(self):
        res = 1.66 * (self._egrid / 1e5)**(-2.6) * 1e-18  # Custom

        return res

    def ice_parser(self, filename):

        store = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader):
                if row_num == 0:
                    continue
                store.append(row[0].split())
        store = np.array(store, dtype=float)

        return store

    # log10(E_nu/GeV)_min, log10(E_nu/GeV)_max, Dec_nu_min[deg], Dec_nu_max[deg], A_Eff[cm^2]

    def smearing_function(self, true_e, true_dec, year):
        """"
        parameters are float

        
        """
        # Returns the smeared reconstructed values
        e_test = true_e
        angle_test = true_dec
        local_smearing = self.smearing_dic[year]
        cross_check_smear_egrid = (local_smearing[:, 1] + local_smearing[:, 0]) / 2.
        idE = np.abs(cross_check_smear_egrid - e_test).argmin()
        all_near_e = (np.where(cross_check_smear_egrid == cross_check_smear_egrid[idE])[0])
        cross_check_smear_theta = (local_smearing[:, 2] + local_smearing[:, 3]) / 2.
        idtheta = np.abs(cross_check_smear_theta - angle_test).argmin()
        all_near_theta = (np.where(cross_check_smear_theta == cross_check_smear_theta[idtheta])[0])
        elements_of_interest = np.intersect1d(all_near_e, all_near_theta)
        tmp_local_smearing = local_smearing[elements_of_interest]
        smearing_e_grid = np.unique(tmp_local_smearing[:, 4])
        smearing_fraction = []

        for smearing_e_loop in smearing_e_grid:
            idE = np.abs(tmp_local_smearing[:, 4] - smearing_e_loop).argmin()
            all_near_e = (np.where(tmp_local_smearing[:, 4] == tmp_local_smearing[:, 4][idE])[0])
            smearing_fraction.append(np.sum(tmp_local_smearing[all_near_e][:, -1]))

        # Normalizing
        smearing_fraction = np.array(smearing_fraction) / np.trapz(smearing_fraction, x=smearing_e_grid)

        return smearing_e_grid, smearing_fraction

    def effective_area_func(self, flux, year):
        """
        Try to make sense of this ----------- 01.09.21 !!!!!!!

        Parameters:
        -------------------------
        surface_fluxes: dic
        year: float

        returns:
        -------------------------
        unsmeared_atmos_counts: dic
        unsmeared_astro_counts: dic
        m_egrid: numpy arra
        eff_areas : dic ------06.09.21

        """
        # Apply the effective area to the simulation and return unsmeared counts

        ch_egrid = (self.eff_dic[year][:, 1] + self.eff_dic[year][:, 0])/2.
        ch_theta = (self.eff_dic[year][:, 2] + self.eff_dic[year][:, 3])/2.
        eff_areas = {}
        unsmeared_astro_counts = {}
        unsmeared_atmos_counts = {}
        for j, theta in enumerate(list(flux.keys())):
            surf_counts = flux[theta][-1]  # should only need to multiply with fluxes
            m_egrid = flux[theta][0]
            
            tmp_eff = []
            check_angle = (theta)
            for energy in m_egrid:
                if energy < 1e1:
                    tmp_eff.append(0.)
                   
                else:

                    loge = np.log10(energy)
                    idE = np.abs(ch_egrid - loge).argmin()
                    all_near = (np.where(ch_egrid == ch_egrid[idE])[0])
                    idTheta = np.abs(ch_theta[all_near] - check_angle).argmin()
                    tmp_eff.append(self.eff_dic[year][all_near, -1][idTheta])
            loc_eff_area = np.array(tmp_eff)


            # print(len(surf_counts))
            # print(len(loc_eff_area))
            # print(len(self.uptime_tot_dic[year]))
            
            tmp_at_un = ((surf_counts *
                            loc_eff_area *
                            self.uptime_tot_dic[year] *
                            flux[theta][1] *
                            2. * np.pi))
            tmp_as_un = ((self.astro_flux() *
                              loc_eff_area *
                              flux[theta][1] *
                              self.uptime_tot_dic[year] *
                              2. * np.pi))
            
            
            unsmeared_atmos_counts[theta] = tmp_at_un
            unsmeared_astro_counts[theta] = tmp_as_un
        return unsmeared_atmos_counts, unsmeared_astro_counts, m_egrid

    def _sim_to_dec(self, flux: dict, year):
        """
        Returns Counts for atmospheric and astro fluxes for IceCube


        """
        # Converts simulation data to detector data
        at_counts_unsm, as_counts_unsm, m_egrid = self.effective_area_func(flux, year)
        log_egrid = np.log10(m_egrid)
        
        self._atmos_counts = {}
        self._astro_counts = {}
        for theta in tqdm((list(flux.keys()))):
            
            check_angle = (theta)
            tmp_1 = []
            tmp_2 = []
            for id_check in range(len(log_egrid)):
                smearing_e, smearing = self.smearing_function(log_egrid[id_check], check_angle, year)
                # print(len(at_counts_unsm[theta]), len(smearing))
                if len(smearing) < 3:, x=range(91)
                    continue
                tmp_1.append(UnivariateSpline(smearing_e,
                                              (smearing *
                                               at_counts_unsm[theta][id_check]),
                                              k=1, s=0,
                                              ext=1)(np.log10(self._egrid)))
                tmp_2.append(UnivariateSpline(smearing_e,
                                              (smearing *
                                               as_counts_unsm[theta][id_check]),
                                              k=1, s=0,
                                              ext=1)(np.log10(self._egrid)))
            self._atmos_counts[theta] = np.sum(tmp_1, axis=0)
            self._astro_counts[theta] = np.sum(tmp_2, axis=0)

        return self._atmos_counts, self._astro_counts

# ------------------------------------------------
# POne funcions -------- ------ -----
    def _simdec_Pone(self, flux: array):
        self.bkgrd_down = {}
        self.bkgrd_horizon = {}
        # backgorund dictionary repositioned
        self._bkgrd = {}

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
                    self.bkgrd_down[i] = (
                        (flux[angle][i] +
                         self.astro_flux(self._egrid)) * self._uptime *
                        self._ewidth * self._aeff.spl_A15(self._egrid)
                    )
                # Horizon
                else:
                    horizon_angles.append(rad)
                    self.bkgrd_horizon[i] = (
                        (flux[angle][i] +
                         self.astro_flux(self._egrid)) * self._uptime *
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
            self._bkgrd[i] += 2. * np.trapz(self.bkgrd_horizon[i][
                                                    sorted_ids],
                                            x=horizon_angles[sorted_ids],
                                            axis=0)
            # Upgoing we assume the same flux for all
            self._bkgrd[i] += (
                (np.pi / 2 - np.pi / 3) *
                (flux[0.][i] +
                 self.astro_flux(self._egrid)) * self._uptime *
                self._ewidth * self._aeff.spl_A51(self._egrid)
            )
            self._bkgrd[i] = self._bkgrd[i] * 46

            return self._bkgrd

    @property
    def egrid(self):
        """ Fetches the calculation egrid

        Returns
        -------
        np.array
            The energy grid
        """
        return self._egrid
