from config import config

import numpy as np
import matplotlib.pyplot as plt
# from limit_calc import Limits
# from bkgrd_calc import Background
# from signal_calc import Signal
from pdm import PDM
import pickle
import time

config["general"]["detector"] = "IceCube"
config['general']["channel"] = "\[Tau]"
#config["simulation parameters"]["low energy cutoff"] = 5e2
ch_name='\[Tau]'
start = time.time()
pdm = PDM()
limits_results = pdm.results
limits_signal_data = pdm.signal
end = time.time()
print("Time taken %.1f" % (start-end))
pickle.dump(limits_results, open("../data/limits_results_gal_%s.pkl" % (ch_name), "wb"))
pickle.dump(limits_signal_data, open("../data/limits_signal_grid_re_gal_%s.pkl"%(ch_name), "wb"))

mass_grid = config['simulation parameters']['mass grid']
sv_grid = config['simulation parameters']['sv grid']
plt.title(r"IceCube Limits DM->$\nu$$\bar{\nu}$")
plt.imshow(limits_results["nue"],
           origin='lower', extent=(min(np.log10(mass_grid)),
                                   max(np.log10(mass_grid)),
                                   min(np.log10(sv_grid)),
                                   max(np.log10(sv_grid))))  # origin!!!!!!!!!
plt.colorbar()
plt.savefig("../data/Limits_result.png")


