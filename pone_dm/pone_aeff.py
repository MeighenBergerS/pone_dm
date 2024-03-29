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
from atm_shower import Atm_Shower
import csv
import pickle
_log = logging.getLogger("pone_dm")


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
        self._shower = Atm_Shower()
        self._egrid = self._shower._egrid  # ###########
        self._ewidth = self._shower._ewidth
        self.days = 60. * 24.
        self.minutes = 60.
        if config["general"]["detector"] == "POne":

            location = config['pone']['aeff location']
            _log.debug('Fetching them from ' + location)
            if config["general"]["pone type"] == "old":
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
                                              self._const.msq2cmsq,
                                              k=1, s=0, ext=3)
                self._A_51 = UnivariateSpline(A_51[:, 0], A_51[:, 1] *
                                              self._const.msq2cmsq,
                                              k=1, s=0, ext=3)
                self._A_55 = UnivariateSpline(A_55[:, 0], A_55[:, 1] *
                                              self._const.msq2cmsq,
                                              k=1, s=0, ext=3)
            elif config["general"]["pone type"] == "new":
                a_eff_file = pickle.load(open(
                                              "../data/aeff_cluster_nuecc.pkl",
                                              'rb'))
                cos_zen_edges, log10E_edges, aeff_hist = a_eff_file
                self._aeff_cos_grid = cos_zen_edges
                self._aeff_e_grid = log10E_edges
                self._aeff_hist = aeff_hist
                self._hit = config['pone_christian']['hit']
                self._module = config['pone_christian']['module']
                self._spacing = config['pone_christian']['spacing']
                self._pos_res = config['pone_christian']['pos res']

        elif config["general"]["detector"] == "IceCube":
            print("Loading Effective Area")
            _log.info("Loading Effective Area for IceCube...")

            self.eff_areas = [
                    '../data/icecube_10year_ps/irfs/IC40_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC59_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC79_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_I_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_II_effectiveArea.csv',
                ]

            self._eff_dic = {
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

        elif config["general"]["detector"] == "combined":
            # PONE effective area
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
                                          self._const.msq2cmsq,
                                          k=1, s=0, ext=3)
            self._A_51 = UnivariateSpline(A_51[:, 0], A_51[:, 1] *
                                          self._const.msq2cmsq,
                                          k=1, s=0, ext=3)
            self._A_55 = UnivariateSpline(A_55[:, 0], A_55[:, 1] *
                                          self._const.msq2cmsq,
                                          k=1, s=0, ext=3)
            print("Loading Effective Area")
            # IceCube effective area
            _log.info("Loading Effective Area for IceCube...")
            self.eff_areas = [
                    '../data/icecube_10year_ps/irfs/IC40_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC59_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC79_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_I_effectiveArea.csv',
                    '../data/icecube_10year_ps/irfs/IC86_II_effectiveArea.csv',
                ]
            self._eff_dic = {
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

    def eff_dic(self):
        return self._eff_dic

#    @property
#    def egrid(self):
#        return self._egrid

#    @property
#    def ewidth(self):
#        return self._ewidth

    def effective_area_func(self, flux: dict, year: float, boolean_sig=False):
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
        _egrid: numpy arra
        eff_areas : dic ------06.09.21

        """
        # Apply the effective area to the simulation and return unsmeared
        # TODO: surf_counts

        ch_egrid = (self._eff_dic[year][:, 1] + self._eff_dic[year][:, 0])/2.
        ch_theta = (self._eff_dic[year][:, 2] + self._eff_dic[year][:, 3])/2.
        unsmeared_astro_counts = {}
        unsmeared_atmos_counts = {}
        eff_area = {}

        for j, theta in enumerate(list(flux.keys())):

            if boolean_sig:
                surf_counts = flux[theta]
            else:
                surf_counts = flux[theta][-1]

            tmp_eff = []
            check_angle = (theta)

            for energy in self._egrid:
                if energy < 1e1:
                    tmp_eff.append(0.)

                else:

                    loge = np.log10(energy)
                    idE = np.abs(ch_egrid - loge).argmin()
                    all_near = (np.where(ch_egrid == ch_egrid[idE])[0])
                    idTheta = np.abs(ch_theta[all_near] - check_angle).argmin()
                    tmp_eff.append(self._eff_dic[year][all_near, -1][idTheta])
            loc_eff_area = np.array(tmp_eff)
            eff_area[theta] = loc_eff_area
            tmp_at_un = ((surf_counts *
                          loc_eff_area *
                          self.uptime_tot_dic[year] *
                          self._ewidth *
                          2. * np.pi))
            tmp_as_un = ((self.astro_flux() *
                          loc_eff_area *
                          self._ewidth *
                          self.uptime_tot_dic[year] *
                          2. * np.pi))

            unsmeared_atmos_counts[theta] = tmp_at_un
            unsmeared_astro_counts[theta] = tmp_as_un
        return unsmeared_atmos_counts, unsmeared_astro_counts, eff_area

    @property
    def aeff_hist(self):
        """"Effective Area histogram
        parameters:
        ---------------------
        histogram with indices :
        hit threshold
        module threshold
        min position resolution
        spacing

        return:
        ---------------------
        aeff_matrix = aeff_hist.loc[ hit, module, min pos res,
                                     spacing]['aeff_hist']
        aeff_matrix [ cos_zen_edges, log10E_edges]

        """
        return self._aeff_hist

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

    def astro_flux(self):
        res = 1.66 * (self._egrid / 1e5)**(-2.6) * 1e-18  # Custom

        return res
