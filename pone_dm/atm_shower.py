# -*- coding: utf-8 -*-
# Name: atm_shower.py
# Authors: Kruteesh Desai, Stephan Meighen-Berger
# Quick and dirty interface to MCEq for the pdm package

# Imports
import logging
import sys
import pickle
from config import config
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
                except ValueError:
                    tmp_particle_store[particle] = (
                        self._mceq_instance.get_solution(particle, 0)
                    )
            # Adding results to dic
            self._particle_fluxes[angle] = tmp_particle_store
        # Dumping for later usage
        pickle.dump([self._egrid, self._ewidth, self._particle_fluxes],
                    open(self._load_str, "wb"))
