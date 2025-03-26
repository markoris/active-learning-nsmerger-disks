import sys
import numpy as np
sys.path.append('../nubhlight/script/analysis')
from torus_id import solve_for_rmax_rho_unit_tabulated as fm_params

param_file = sys.argv[1]

eospath = '../nubhlight/data/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5'

params = np.loadtxt(param_file)[:, 1:]

for param in params:
	try:
		print('---')
		print(param)
		print('---')	
		torus_params = np.array(fm_params(param[0], param[1], param[2], eospath, param[4], param[3]))
		if np.isnan(torus_params).any(): continue
		try:
			input_params = np.concatenate((input_params, param[None, :]), axis=0)
			fmparams = np.concatenate((fmparams, torus_params[None, :]), axis=0)
		except NameError:
			input_params = param[None, :]
			fmparams = torus_params[None, :]
	except ValueError as err:
		if 'Invalid Adiabat' in err.args[0]:
			print('Invalid adiabat for parameters ', param)
			continue
		else: 
			print(err)
			continue
	#if fmparams.shape[0] == 5: break # limit to 5 simulations placed at a time, removed since 1-2 have problems each batch
	del torus_params

np.savetxt('torus_parameters.dat', np.c_[input_params, fmparams], fmt='%1.4f')
print(params.shape)
print(fmparams.shape)
