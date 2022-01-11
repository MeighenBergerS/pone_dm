from config import config
# from pone_aeff import Aeff
# from dm2nu import DM2Nu
# from atm_shower import Atm_Shower
# from detectors import Detector
import numpy as np
import matplotlib.pyplot as plt
# from limit_calc import Limits
# from bkgrd_calc import Background
# from signal_calc import Signal
from pdm import PDM


config["general"]["detector"] = "IceCube"
pdm = PDM()
limits_results = pdm.results['numu']
mass_grid = config['simulation parameters']['mass grid']
sv_grid = config['simulation parameters']['sv grid']
plt.title(r"IceCube Limits DM->$\nu$$\bar{\nu}$")
plt.imshow(limits_results,
           origin='lower', extent=(min(np.log10(mass_grid)),
                                   max(np.log10(mass_grid)),
                                   min(np.log10(sv_grid)),
                                   max(np.log10(sv_grid))))  # origin!!!!!!!!!
plt.colorbar()
plt.savefig("Limits_result.png")
