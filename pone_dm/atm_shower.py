# -*- coding: utf-8 -*-
# Name: atm_shower.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Quick and dirty interface to MCEq for the pdm package

# Imports
import logging
import sys
import pickle
from .config import config
import numpy as np
# Imports
from tqdm import tqdm
import csv
from iminuit import Minuit
from scipy.interpolate import UnivariateSpline
_log = logging.getLogger(__name__)


class Atm_Shower(object):
    """ Class to interface with the MCEq package. This is just
    a quick and dirty approach

    Parameters
    ----------
    None

    Returns
    -------
    None

    Errors
    ------
    ValueError
        Unknown settings for MCEq
    """

    def __init__(self):

        if config["general"]["detector"] == "POne":
            self._load_str = config["advanced"]["atmospheric storage"] + "shower.p"
            self.particle_name = ["numu", "nue", "nutau"]
            try:
                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for " + self._load_str)

                loaded_atm = pickle.load(open(self._load_str, "rb"))
                print("ayay")
                self._egrid = loaded_atm[0]
                self._ewidth = loaded_atm[1]
                self._particle_fluxes = loaded_atm[2]
            except FileNotFoundError:
                _log.info("Failed to load data. Generating..")
                # Importing the primary flux models
                _log.info('Importing Primary models')
                import crflux.models as pm
                # Checking if MCEq is native or self built
                _log.info('Importing MCEq')
                if config['atmospheric showers']['native mceq']:
                    _log.debug('Using a native MCEq')
                    from MCEq.core import MCEqRun
                else:
                    _log.debug('Using a custom MCEq')
                    sys.path.insert(0,
                                    config['atmospheric showers']['path to mceq'])
                    from MCEq.core import MCEqRun
                # Setting options
                _log.info('Setting MCEq options')
                self._atmosphere = config['atmospheric showers']['atmosphere']
                self._i_model = config['atmospheric showers']['interaction model']
                if config['atmospheric showers']['primary model'] == 'H4a':
                    self._p_model = (pm.HillasGaisser2012, 'H4a')
                else:
                    ValueError('Unsupported primary model!' +
                            ' Please check the config file')
                    sys.exit()
                _log.info('initializing a MCEq instance')
                self._mceq_instance = MCEqRun(
                    interaction_model=self._i_model,
                    primary_model=self._p_model,
                    theta_deg=0,
                )
                self._egrid = self._mceq_instance.e_grid
                self._ewidth = self._mceq_instance.e_widths
                self._mceq_instance.set_density_model(self._atmosphere)
                # Running simulations
                self._run_mceq()

        elif config["generral"]["detector"] == "IceCube":
            self.minutes = 60.
            self.days = 60. * 24
            # Loading simulation results
            self.surface_fluxes = pickle.load(open("../data/surf_store_v1.p", "rb"))
            # Adding 90 deg
            self.surface_fluxes[90] = self.surface_fluxes[89]

            self._load_str = config["advanced"]["atmospheric storage"] + "atmos_all.p"

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
            # MJD, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg], Azimuth[deg], Zenith[deg]
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

            try:
                _log.info("Trying to load pre-calculated tables")
                _log.debug("Searching for " + self._load_str)
                loaded_atm = pickle.load(open(self._load_str, "rb"))
                print("ayay IceCube")
                self._egrid = loaded_atm[0]
                self._ewidth = loaded_atm[1]
                self._particle_fluxes = loaded_atm[2]
            except FileNotFoundError:
                _log.info("calculating fluxes")

                self.atmos_all = {}
                self.astro_all = {}
                for year in range(10):
                    self.particle_counts_smeared_unin, self.astro_counts_smeared_unin, self.m_egrid_smeared = self.sim_to_dec(self.surface_fluxes, year)
                    self.particle_counts_smeared = np.trapz(self.particle_counts_smeared_unin, x=list(self.surface_fluxes.keys()), axis=0)
                    self.astro_counts_smeared = np.trapz(self.astro_counts_smeared_unin, x=list(self.surface_fluxes.keys()), axis=0)
                    self.atmos_all[year] = self.particle_counts_smeared
                    self.astro_all[year] = self.astro_counts_smeared
                pickle.dump(self.astro_all, open('../data/astro_all.p', 'wb'))
                pickle.dump(self.atmos_all, open('../data/atmos_all.p', 'wb'))

            self.uptime_tot_dic = {}

            _log.info(" filtering counts")

            for year in range(10):
                self.uptime_tot_dic[year] = np.sum(np.diff(self.uptime_dic[
                                                self.year])) * self.days

            self.filtered_data = self.data_filter(self.event_dic, [0., 9.], [0., 90.], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            self.total_data, _ = np.histogram(self.filtered_data[0][:,1], bins=np.log10(self.surface_fluxes[85][2]))
            self.total_atmos = self.atmos_all[0]
            self.total_astro = self.astro_all[0]
            for self.year in range(10):
                if year == 0:
                    continue
                tmp, _ = np.histogram(self.filtered_data[year][:, 1], bins=np.log10(self.surface_fluxes[85][2]))
                self.total_data += tmp
                self.total_atmos += self.atmos_all[year]
                self.total_astro += self.astro_all[year]
            _log.info("Fitting procedure of counts")
            self.fit_range_low = 51
            self.fit_range_high = 71

            self.sigma_y = np.ones_like(self, self._egrid)[self.fit_range_low:self.fit_range_high] * 0.2

            self.m = Minuit(self.LSQ, 0.5, 0.5, 0.1, 0.1)
            self.m.limits['norm_atmos'] = (0.5, 100.)
            self.m.limits['norm_astro'] = (0.9, 2.)
            self.m.limits['shift_atmos'] = (-1, 1.)
            self.m.limits['shift_astro'] = (-0.1, 0.1)
            self.m.migrad()
            # self.spl_atmos = UnivariateSpline(np.log10(self._egrid) + self.m.values[2],
              #              self.total_atmos * self.m.values[0], k=1, s=0)
            # self.spl_astro = UnivariateSpline(np.log10(self.m_egrid) + self.m.values[3],
              #              self.total_astro * self.m.values[1], k=1, s=0)
            
            #   Trying to omit fluxes atmospheric fluxes----------------31.08.21

    @property
    def astro_flux(self):
        res = 1.66 * (self._egrid / 1e5)**(-2.6) * 1e-18  # Custom
        return res

    def ice_parser(filename):
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

    def effective_area_func(self, surface_fluxes, year):
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

        """
        # Apply the effective area to the simulation and return unsmeared counts

        cross_check_egrid = (self.eff_dic[year][:, 1] + self.eff_dic[year][:, 0])/2.
        cross_check_theta = (self.eff_dic[year][:, 2] + self.eff_dic[year][:, 3])/2.

        for theta in list(surface_fluxes.keys()):
            surf_counts = surface_fluxes[0][-1]  # should only need to multiply with fluxes
            m_egrid = surface_fluxes[0][0]
            eff_areas = []
            unsmeared_atmos_counts = {}
            unsmeared_astro_counts = {}
            check_angle = (theta)
            for energy in m_egrid:
                if energy < 1e1:
                    eff_areas.append(0.)
                else:
                    loge = np.log10(energy)
                    idE = np.abs(cross_check_egrid - loge).argmin()
                    all_near = (np.where(cross_check_egrid == cross_check_egrid[idE])[0])
                    idTheta = np.abs(cross_check_theta[all_near] - check_angle).argmin()
                    eff_areas.append(self.eff_dic[year][all_near, -1][idTheta])
            loc_eff_area = np.array(eff_areas)
            unsmeared_atmos_counts[year, theta] = surf_counts * loc_eff_area * self.uptime_tot_dic[year] * surface_fluxes[theta][1] * 2. * np.pi
            unsmeared_astro_counts[year, theta] = (
                self.astro_flux(m_egrid) * loc_eff_area * surface_fluxes[theta][1] * self.uptime_tot_dic[year] * 2. * np.pi
            )

        return unsmeared_atmos_counts, unsmeared_astro_counts, m_egrid

    def sim_to_dec(self, surface_fluxes, year):
        """
        Try to make sense of this -------01.09.21 !!!!!!!!!

        """
        # Converts simulation data to detector data
        atmos_counts_unsmeared, astro_counts_unsmeared, m_egrid = self.effective_area_func(surface_fluxes, year)
        log_egrid = np.log10(m_egrid)
        smeared_atmos = []
        smeared_astro = []
        self.atmos_spl = {}
        self.astro_spl = {}
        for theta in tqdm((list(self.surface_fluxes.keys()))):
            check_angle = (theta)
            smeared_atmos_loc = []
            smeared_astro_loc = []
            int_grid = []
            for id_check in range(len(log_egrid)):
                smearing_e, smearing = self.smearing_function(log_egrid[id_check], check_angle, year)
                if len(smearing) < 3:
                    continue
                self.atmos_spl[year, theta] = UnivariateSpline(smearing_e, smearing * atmos_counts_unsmeared[year, theta][id_check],
                                            k=1, s=0, ext=1)
                self.astro_spl[year, theta] = UnivariateSpline(smearing_e, smearing * astro_counts_unsmeared[year, theta][id_check],
                                            k=1, s=0, ext=1)
                smeared_atmos_loc.append(self.atmos_spl[year, theta](log_egrid))
                smeared_astro_loc.append(self.astro_spl[year, theta](log_egrid))
                int_grid.append(log_egrid[id_check])
            smeared_atmos.append(np.trapz(smeared_atmos_loc, x=int_grid, axis=0))
            smeared_astro.append(np.trapz(smeared_astro_loc, x=int_grid, axis=0))
        return np.array(smeared_atmos), np.array(smeared_astro), m_egrid

    def data_filter(event_dic, energy_range, angle_range, years):
        # filters the data in energy and angle
        filtered_dic = {}
        for year in years:
            # where method is faster as basic slicing
            energy_filter_1 = event_dic[year][np.where(event_dic[year][:, 1] < energy_range[1])] 
            energy_filter_2 = energy_filter_1[np.where(energy_filter_1[:, 1] > energy_range[0])]
            high_angle = angle_range[1]
            angle_filter_1 = energy_filter_2[np.where(energy_filter_2[:, 4] < high_angle)]
            low_angle = angle_range[0]
            angle_filter_2 = angle_filter_1[np.where(angle_filter_1[:, 4] > low_angle)]
            filtered_dic[year] = angle_filter_2
        return filtered_dic

    def background_fit(self, norm_atmos, norm_astro, shift_atmos, shift_astro):
                    spl_atmos = UnivariateSpline(np.log10(self._egrid) + shift_atmos,
                                                 self.total_atmos * norm_atmos, k=1, s=0)
                    spl_astro = UnivariateSpline(np.log10(self._egrid) + shift_astro,
                                                 self.total_astro * norm_astro, k=1, s=0)
                    return np.abs(spl_atmos(np.log10(self._egrid)) + spl_astro(np.log10(self._egrid)))

    def LSQ(self, norm_atmos, norm_astro, shift_atmos, shift_astro):
        ym = self.background_fit(norm_atmos, norm_astro, shift_atmos, shift_astro)[self.fit_range_low:self.fit_range_high]
        res = np.sum((np.log10(self.total_data[self.fit_range_low:self.fit_range_high]+1) -
                     np.log10(ym+1)) ** 2 / self.sigma_y ** 2)
        return res


    @property
    def egrid(self):
        """ Fetches the calculation egrid

        Returns
        -------
        np.array
            The energy grid
        """
        return self._egrid

    @property
    def ewidth(self):
        """ Fetches the calculation e widths

        Returns
        -------
        np.array
            The energy grid widths
        """
        return self._ewidth

    @property
    def flux_results(self) -> dict:
        """ Fetches the particle flux dictionaries

        Returns
        -------
        dict
            The flux results at the desired angles and the desired
            particles.
        """
        return self._particle_fluxes

    def _run_mceq(self):
        """ Runs the mceq simulation with the set parameters
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        # Storing simulation results here
        self._particle_fluxes = {}
        _log.info('Running MCEq simulations')
        _log.debug('Running %d simulations for the angles' % (
            len(config['atmospheric showers']['theta angles']))
        )
        for angle in config['atmospheric showers']['theta angles']:
            # Temporary storage
            tmp_particle_store = {}
            _log.debug('Currently at angle %.2f' % angle)
            # Setting the angles
            _log.debug('Setting the angle')
            self._mceq_instance.set_theta_deg(angle)
            _log.debug('Running the simulation')
            self._mceq_instance.solve()
            # Fetching totals. If not using the explicit version
            for particle in (
                             config['atmospheric' +
                                    ' showers']['particles of interest']):
                # Trying totals
                try:
                    tmp_particle_store[particle] = (
                        self._mceq_instance.get_solution('total_' + particle,
                                                         0)
                    )
                # Falling back to explicit version
                except:
                    tmp_particle_store[particle] = (
                        self._mceq_instance.get_solution(particle, 0)
                    )
            # Adding results to dic
            self._particle_fluxes[angle] = tmp_particle_store
        # Dumping for later usage
        pickle.dump([self._egrid, self._ewidth, self._particle_fluxes],
                    open(self._load_str, "wb"))
