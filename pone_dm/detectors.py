# -*- coding: utf-8 -*-
# Name: IceCube_extraction.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Quick and dirty interface to MCEq for the pdm package

# Imports
# from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
import logging
from pone_aeff import Aeff
from config import config
from constants import pdm_constants
import numpy as np
from tqdm import tqdm
import csv
import pickle
from scipy.interpolate import UnivariateSpline
from scipy.odr import ODR, Model, Data, RealData
_log = logging.getLogger("pone_dm")


class Detector(object):
    """
    Class to extract counts depending on the type of Detectors
    IceCube and POne are only two choices there are as of now---------

    """

    def __init__(self, aeff: Aeff):

        self.name = config["general"]["detector"]
        self._aeff = aeff
        self._const = pdm_constants()
        self._egrid = self._aeff._egrid
        self._ewidth = self._aeff._ewidth
        self._particles = (
            config['atmospheric showers']['particles of interest']
        )
        self._uptime = config['simulation parameters']['uptime']
        self.days = 60. * 24
        self.name = config["general"]["detector"]

        if self.name == "IceCube":
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
    # MJ    D, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg],
    # Az    imuth[deg], Zenith[deg]

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
            self._sim2dec = self.sim_to_dec

        elif self.name == "POne":
            if config["pone"]['smearing'] == 'smearing':
                self._low_sigma = config['pone']['low E sigma']
                self._high_sigma = config['pone']['high E sigma']
                self.spl_mid_mean = {}
                self.spl_mid_sigma = {}
                for p in config['atmospheric showers'][
                                'particles of interest']:
                    self.spl_mid_mean[p] = UnivariateSpline([1e3, 1e4],
                                                            [700., 1e4],
                                                            k=1)
                    self.spl_mid_sigma[p] = UnivariateSpline([1e3, 1e4], [
                        self._low_sigma[p], self._high_sigma[p]], k=1)
            if config['general']['pone type'] == 'old':
                self._sim2dec = self.simdec_Pone
            elif config['general']['pone type'] == 'new':
                self._sim2dec = self.simdec_Pone_new

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

        at_counts_unsm, as_counts_unsm, eff_area = (
            self._aeff.effective_area_func(_flux,
                                           year,
                                           boolean_sig)
        )
        log_egrid = np.log10(self._egrid)
        self._count = {}
        self._tmp_count = []

        self._counts_at_eff = at_counts_unsm
        self._counts_as_eff = as_counts_unsm

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
            # appending array to a list ( tmp_1(e_bin)_theta )s

            if boolean_sig is False:
                self._tmp_count.append(np.sum(np.add(np.array(tmp_1),
                                                     np.array(tmp_2)),
                                              axis=0))
            else:
                self._tmp_count.append(np.sum(np.array(tmp_1),
                                              axis=0))

        # suming up for all the angles ------
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
        flux : dict ( label : [angle][flavour])

        returns
        ------------------
        _count : Dict ( label : neutrino flavour)
                [Total background ( astro + atmos )]

        """
        self.count_down = {}
        self.count_horizon = {}
        # backgorund dictionary repositioned
        self._count = {}
        thetas = np.array([i for i in flux.keys()])
        # Differentiation between Background and signal Counts conversion
        if boolean_sig:
            Astro = np.zeros_like(self.astro_flux())
        else:
            Astro = np.array(self.astro_flux())

        for i in self._particles:
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
                    local_log_norm = (self._log_norm(self._egrid,
                                                     mu[i], sigma[i]))
                    tmp_count_mat.append(self._count[i][k] * local_log_norm)
                ratio = (np.sum(self._count[i]) /
                         np.sum(np.array(np.sum(tmp_count_mat, axis=0))))
                tmp_count_mat_r = []
                for k, e in enumerate(self._egrid):
                    mu, sigma = self._distro_parms(e)

                    local_log_norm = self._log_norm(self._egrid, mu[i],
                                                    sigma[i])
                    tmp_count_mat_r.append(self._count[i][k] *
                                           local_log_norm * ratio)

                self._count[i] = np.array(np.sum(tmp_count_mat_r, axis=0))

        return self._count

# ------------------------------------------------------
# P-ONE new effective areas
# add smearing function for new P-ONE !!!!!
    def simdec_Pone_new(self, flux: dict, boolean_sig=False,
                        boolean_smeared=False):
        """Converts the fluxes to counts with the Chriostian's effective
        areas.

        paramters:
        ----------------------
        flux: Dictionary [ angle] [flavour]
        boolean_sig: bool , if signal fluxes are being calculated then True
        booolean_smeared: bool, if smearing is needed to be included

        return:
        ----------------------
        counts: Dictionary, label:[ Neutrino Flavour] ---> each  flavours has
                dimension [e_grid]
        counts_ang: dictionray, label:[Neutrino flavour] ---> each flavours
                    has dimensions [ angle_grid x e_grid ]

        """
        self._hit = config['pone_christian']['hit']
        self._module = config['pone_christian']['module']
        self._spacing = config['pone_christian']['spacing']
        self._pos_res = config['pone_christian']['pos res']
        if type(self._spacing) is not list:
            zen_thetas = self._aeff._aeff_cos_grid
            aeff_log10e = self._aeff._aeff_e_grid
            zen_grid = self.width2grid(zen_thetas)
            aeff_log_e_grid = self.width2grid(aeff_log10e)
            aeff_mat = (
             self._aeff.aeff_hist.loc[self._hit, self._module,
                                      self._pos_res,
                                      self._spacing]["aeff_hist"]
            )

            if boolean_sig:
                Astro = np.zeros_like(self.astro_flux())
            else:
                Astro = np.array(self.astro_flux())
            counts = {}
            counts_ang = {}
            rad = np.arccos(zen_grid)
            for p in self._particles:
                tmp_counts = []
                angles = flux.keys()
                for i_t, theta in tqdm(enumerate(angles)):
                    spl_aeff = UnivariateSpline(aeff_log_e_grid,
                                                (self._const.msq2cmsq *
                                                 aeff_mat[i_t]),
                                                s=0, k=1)
                    aeff_eval = spl_aeff(np.log10(self._egrid))
                    aeff_eval[self._egrid < 10**min(aeff_log_e_grid)] = 0
                    aeff_eval[self._egrid > 10**max(aeff_log_e_grid)] = 0
                    tmp_counts.append(aeff_eval *
                                      (flux[theta][p] + Astro) *
                                      self._uptime *
                                      rad[i_t] *
                                      self._ewidth *
                                      2  # Since the theta angles are not
                                         # allowed all the way through but
                                         #  we have symmetry
                                      )
                counts_ang[p] = tmp_counts
                counts[p] = np.trapz(counts_ang[p],
                                     x=np.array([i for i in flux.keys()]),
                                     axis=0)
        if boolean_smeared:
                smearing_file = pickle.load(open("../data/fisher_casc.pkl",'rb'))
                energies = smearing_file.loc[("Full pessimistic cluster", 50, slice(None), 16), "logE"].index.get_level_values(2)
                log10Esigmas = smearing_file.loc[("Full pessimistic cluster", 50, slice(None), 16), "logE"].values

                tmp_count_mat = []
                ratio = []
                for k, e in enumerate(self._egrid):
                    mu, sigma = self._distro_parms(e)
                    local_log_norm = (self._log_norm(self._egrid,
                                                     mu[p], sigma[p]))
                    tmp_count_mat.append(self._count[p][k] * local_log_norm)
                ratio = (np.sum(self._count[p]) /
                         np.sum(np.array(np.sum(tmp_count_mat, axis=0))))
                tmp_count_mat_r = []
                for k, e in enumerate(self._egrid):
                    mu, sigma = self._distro_parms(e)

                    local_log_norm = self._log_norm(self._egrid, mu[p],
                                                    sigma[p])
                    tmp_count_mat_r.append(self._count[p][k] *
                                           local_log_norm * ratio)

                self._count[i] = np.array(np.sum(tmp_count_mat_r, axis=0))

        return counts, counts_ang

    def width2grid(self, a: np.array):
        m_a = []
        for i, e in enumerate(a):
            if i == 0:
                m_a.append(a[i])
            elif i == len(a)-1:
                break
            else:
                m_a.append((a[i] + a[i+1]) / 2)
        return m_a



