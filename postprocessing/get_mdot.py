import sys, glob
sys.path.append('/users/mristic/jonah_sims/nubhlight/script/analysis/')
import numpy as np
import hdf5_to_dict
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

sim_dirs = np.array(glob.glob('../mbh*'))
sim_dirs = np.sort(sim_dirs)

for sim in sim_dirs:
	sim_dict = hdf5_to_dict.load_diag(sim+'/dumps')
	window_size = 21
	mdot_smooth = savgol_filter(sim_dict['mdot'], window_size, 2)
	plt.plot(sim_dict['mdot'], color='k', label='original')
	plt.plot(mdot_smooth, color='red', label='smoothed')
	plt.xscale('log')
	plt.legend()
	plt.savefig(sim+'/dumps/mdot.pdf')
	plt.close()
	mdot = np.min(mdot_smooth)
	print(sim, '\t',  mdot)

