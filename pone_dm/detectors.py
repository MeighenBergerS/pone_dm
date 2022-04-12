# -*- coding: utf-8 -*-
# Name: IceCube_extraction.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Quick and dirty interface to MCEq for the pdm package

# Imports
import logging
from pone_aeff import Aeff
from config import config
import numpy as np
from tqdm import tqdm
import csv
import pickle
from scipy.interpolate import UnivariateSpline
_log = logging.getLogger("pone_dm")


class Detector(object):
    """
    Class to extract counts depending on the type of Detectors
    IceCube and POne are only two choices there are as of now---------

    """

    def __init__(self, aeff: Aeff):

        self.name = config["general"]["detector"]
        self._aeff = aeff
        self._egrid = self._aeff._egrid
        self._ewidth = self._aeff._ewidth

        self._uptime = config['simulation parameters']['uptime']
        self.days = 60. * 24

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
    # MJD, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg],
    # Azimuth[deg], Zenith[deg]

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

            self.uptime_tot_dic[year] = (np.sum(np.diff(
                                                        self.uptime_dic[
                                                            year])) *
                                         self.days)
        self.name = config["general"]["detector"]
        self._low_sigma = config['pone']['low E sigma']
        self._high_sigma = config['pone']['high E sigma']
        self.spl_mid_mean = {}
        self.spl_mid_sigma = {}
        for p in config['atmospheric showers']['particles of interest']:
            self.spl_mid_mean[p] = UnivariateSpline([1e3, 1e4], [700., 1e4],
                                                    k=1)
            self.spl_mid_sigma[p] = UnivariateSpline([1e3, 1e4], [
                self._low_sigma[p], self._high_sigma[p]], k=1)
        if self.name == "IceCube":
            self._sim2dec = self.sim_to_dec
        elif self.name == "POne":
            if config['general']['pone type'] == 'old':
                self._sim2dec = self.simdec_Pone
            elif config['general']['pone type'] == 'new':
                self.sim2dec = self.simdec_Pone_new
        elif self.name == 'combined':
            self._sim2dec_pone = self.simdec_Pone
            self._sim2dec_ice = self.sim_to_dec

    @property
    def sim2dec_pone(self):
        return self._sim2dec_pone

    @property
    def sim2dec_ice(self):
        return self._sim2dec_ice

    @property
    def sim2dec(self):
        return self._sim2dec
# ------------------------------------------
# Icecube functions ------

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

    # log10(E_nu/GeV)_min, log10(E_nu/GeV)_max, Dec_nu_min[deg],
    # Dec_nu_max[deg], A_Eff[cm^2]

    def smearing_function(self, true_e, true_dec, year):
        """"
        parameters:
        tru_e : log_10( E )
        true_dec : Declanation
        year : year in interset

        return:
        smearing_e : log_10(E)
        smearing_fraction : smearing distribution for true over log_10( E )
        grid
        """
        # Returns the smeared reconstructed values
        e_test = true_e
        angle_test = true_dec
        local_smearing = self.smearing_dic[year]
        ch_egrid = (local_smearing[:, 1] + local_smearing[:, 0]) / 2.
        idE = np.abs(ch_egrid - e_test).argmin()
        all_near_e = (np.where(
                               ch_egrid == ch_egrid[idE])[0])
        ch_theta = (local_smearing[:, 2] + local_smearing[:, 3]) / 2.
        idtheta = np.abs(ch_theta - angle_test).argmin()
        all_near_theta = (np.where(ch_theta == ch_theta[idtheta])[0])
        elements_of_interest = np.intersect1d(all_near_e, all_near_theta)
        tmp_sme = local_smearing[elements_of_interest]
        smearing_e_grid = np.unique(tmp_sme[:, 4])
        smearing_fraction = []

        for smearing_e_loop in smearing_e_grid:
            idE = np.abs(tmp_sme[:, 4] - smearing_e_loop).argmin()
            all_near_e = (np.where(tmp_sme[:, 4] == tmp_sme[:, 4][idE])[0])
            smearing_fraction.append(np.sum(tmp_sme[all_near_e][:, -1]))

        # Normalizing
        smearing_fraction = (np.array(smearing_fraction) /
                             np.trapz(smearing_fraction,
                                      x=smearing_e_grid)) / 10  # For change
        # in basis from log_10 to natural number Does this make sense ???????

        return smearing_e_grid, smearing_fraction

    def sim_to_dec(self, flux: dict, year: float):
        """
        Returns Counts for atmospheric and astro fluxes for IceCube --> dict
        parameters
        ----------------s
        flux : Dict [ label : angle ]
        year : float

        Returns
        ----------------
        _count : dict [label : neutrino flavour]
                [Total counts ( atmos + astro )]  sumed over all thetas
        """
        # Converts simulation data to detector data
        if type(flux) != dict:
            _flux = {}
            boolean_sig = True
            for theta in config["atmospheric showers"]["theta angles"]:
                _flux[theta] = flux
        else:
            _flux = flux
            boolean_sig = False

        at_counts_unsm, as_counts_unsm, eff_area = self._aeff.effective_area_func(
            _flux, year, boolean_sig)
        log_egrid = np.log10(self._egrid)
        self._count = {}
        self._tmp_count = []

        self._counts_at_eff = at_counts_unsm
        self._counts_as_eff = as_counts_unsm
        pickle.dump(eff_area, open('../data/eff_area_ice.pkl','wb'))
        if boolean_sig is False:
            pickle.dump(at_counts_unsm,
                        open('../data/counts_unsme/atmo_%f.pkl' % (year),
                             'wb'))
            pickle.dump(as_counts_unsm,
                        open('../data/counts_unsme/atmo_%f.pkl' % (year),
                             'wb'))

        for theta in tqdm(list(_flux.keys())):

            check_angle = (theta)
            tmp_1 = []
            tmp_2 = []
            for id_check in range(len(log_egrid)):
                smearing_e, smearing = self.smearing_function(
                                                            log_egrid[
                                                                id_check],
                                                            check_angle,
                                                            year)
                # print(len(at_counts_unsm[theta]), len(smearing))
                if len(smearing) < 3:
                    continue
                local_sp = UnivariateSpline(smearing_e,
                                            smearing,
                                            k=1, s=0,
                                            ext=1)(np.log10(self._egrid))

                tmp_1.append(local_sp * at_counts_unsm[theta][id_check])
                tmp_2.append(local_sp * as_counts_unsm[theta][id_check])
            # appending array to a list ( tmp_1(e_bin)_theta ) 
            # !!!!!!!!!!!!!!!! Here we are not adding the Astro Physical Fluxes
            #  should be added as I understand !!!!!!!!!!!
            # tmp_c = np.add(np.array(tmp_1), np.array(tmp_2))
            # self._tmp_count.append(np.sum(tmp_c, axis=0))
            self._tmp_count.append(np.sum(np.array(tmp_1), axis=0))
        # suming up for all the angles ------ need to check -----
        self._tmp_count = np.sum(self._tmp_count, axis=0)

        for i in config['atmospheric showers']['particles of interest']:
            # Assuming the same counts for all flavours For IceCube ----------
            self._count[i] = self._tmp_count

        return self._count

# ------------------------------------------------
# POne funcions -------- ------ -----
    def _distro_parms(self, Etrue):
        """ Parameter estimation function depending  on the E_true
        return:
        mu, sigma : tuple
        """
        mu = {}
        sigma = {}
        for p in config['atmospheric showers']['particles of interest']:
            if Etrue < 1e3:
                mu[p] = np.log(700)
                sigma[p] = self._low_sigma[p]
            elif 1e3 <= Etrue <= 1e4:
                mu[p] = np.log(self.spl_mid_mean[p](Etrue))
                sigma[p] = self.spl_mid_sigma[p](Etrue) * np.log(Etrue)
            else:
                mu[p] = np.log(Etrue)
                sigma[p] = self._high_sigma[p]
        return mu, sigma

    def _log_norm(self, E, mu, sigma):
        """Distribution function
        x = E_grid
        mu = log(E)
        sigma = fraction of E * E ( so no fraction or percentage )
        ( standard deviation as per definition )
        """
        pdf = (np.exp(- (np.log(E) - mu)**2 / (2 * sigma**2)) /
               (E * sigma * np.sqrt(2 * np.pi)))

        return pdf

    def simdec_Pone(self, flux: dict, boolean_sig=False,
                    boolean_smeared=False):
        """
        Returns particle counts for P-One Detector
        parameter
        ------------------
        flux : dict ( label : angle: flavour)

        returns
        ------------------
        _count : Dict ( label : neutrino flavour)
                [Total background ( astro + atmos )]

        """
        self.count_down = {}
        self.count_horizon = {}
        # backgorund dictionary repositioned
        self._count = {}

        #  From here the bakgound flux has many thetas whereas for signal
        # we have dfined it differently:::: ??????

        thetas = np.array([i for i in flux.keys()])
        # Differentiation between Background and signal Counts conversion
        if boolean_sig:
            Astro = np.zeros_like(self.astro_flux())
        else:
            Astro = np.array(self.astro_flux())

        for i in (config['atmospheric showers']['particles of interest']):
            self.count_down[i] = []
            down_angles = []
            self.count_horizon[i] = []
            horizon_angles = []
            # Astrophysical is the same everywhere
            for angle in flux.keys():
                rad = np.deg2rad(np.abs(angle - 90.))
                # Downgoing
                if np.pi / 3 <= rad <= np.pi / 2:
                    down_angles.append(rad)
                    self.count_down[i].append(
                        (flux[angle][i] +
                         Astro) * self._uptime *
                        self._ewidth * self._aeff.spl_A15(self._egrid)
                    )
                # Horizon
                else:
                    horizon_angles.append(rad)
                    self.count_horizon[i].append(
                        (flux[angle][i] +
                         Astro) * self._uptime *
                        self._ewidth * self._aeff.spl_A55(self._egrid)
                    )
            # Converting to numpy arrays
            self.count_down[i] = np.array(self.count_down[i])
            self.count_horizon[i] = np.array(self.count_horizon[i])
            down_angles = np.array(down_angles)
            horizon_angles = np.array(horizon_angles)
            # Integrating
            self._count[i] = np.zeros_like(self._egrid)
            sorted_ids = np.argsort(down_angles)
            # Downgoing
            self._count[i] += np.trapz(self.count_down[i][sorted_ids],
                                       x=down_angles[sorted_ids], axis=0)
            # Horizon we assume it is mirrored
            sorted_ids = np.argsort(horizon_angles)
            self._count[i] += 2. * np.trapz(self.count_horizon[i][
                                                    sorted_ids],
                                            x=horizon_angles[sorted_ids],
                                            axis=0)
            # Upgoing we assume the same flux for all
            self._count[i] += (
                (np.pi / 2 - np.pi / 3) *
                (flux[thetas[2]][i] +
                 Astro) * self._uptime *
                self._ewidth * self._aeff.spl_A51(self._egrid)
            )
            self._count[i] = self._count[i]

            if boolean_smeared:
                tmp_count_mat = []
                ratio = []
                for k, e in enumerate(self._egrid):
                    mu, sigma = self._distro_parms(e)
                    # print('mu = %.1e, sigma = %.1e' % (mu[i], sigma[i]))
                    local_log_norm = (self._log_norm(self._egrid,
                                                     mu[i], sigma[i]))
                    tmp_count_mat.append(self._count[i][k] * local_log_norm)
                # print('old_counts = %.1e,  new_counts = %.1e' % (np.sum(
                #    self._count[i]), np.sum(np.array(np.sum(tmp_count_mat,
                #                                            axis=0)))))
                ratio = (np.sum(self._count[i]) /
                         np.sum(np.array(np.sum(tmp_count_mat, axis=0))))
                tmp_count_mat_r = []
                for k, e in enumerate(self._egrid):
                    mu, sigma = self._distro_parms(e)

                    local_log_norm = self._log_norm(self._egrid, mu[i],
                                                    sigma[i])
                    tmp_count_mat_r.append(self._count[i][k] *
                                           local_log_norm * ratio)
                # pickle.dump(ratio, open('../data/smearing_ratio_pone.pkl',
                #  'wb'))
                # print('ratio = %.1e' % (ratio))
                # print('old_counts = %.1e,  new_counts = %.1e' % (np.sum(
                #    self._count[i]), np.sum(np.array(np.sum(tmp_count_mat_r,
                #                                            axis=0)))))
                # suming over the the energy bin vertically to get combined
                # value of counts after smearing over all energy bins
                self._count[i] = np.array(np.sum(tmp_count_mat_r, axis=0))
        pickle.dump(self._aeff.spl_A15(self._egrid), open(
            '../data/eff_area_15.pkl', 'wb'))
        pickle.dump(self._aeff.spl_A51(self._egrid), open(
            '../data/eff_area_51.pkl', 'wb'))
        pickle.dump(self._aeff.spl_A55(self._egrid), open(
            '../data/eff_area_55.pkl', 'wb'))
        return self._count

# ------------------------------------------------------
# P-ONE new effective areas
# add smearing function for new P-ONE !!!!!
    def sim2dec_Pone_new(self):
        return 0
