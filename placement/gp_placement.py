import sys
import numpy as np
import glob
import save_sklearn_gp as ssg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode, rv_histogram
from plotutils.bounded_kde import Bounded_kde as bkde

n_placed = int(sys.argv[1]) # command-line argument specifying how many parameter combinations to identify (how many simulations we want to place this batch)

def placement_params(files, n_placed, n_draws=10000):

	spin_lower_limit = 0.6 # artifically fixing lower bound for spins to 0.6 until further notice

	mins = np.min(data[:, :5], axis=0) # get minima of 5 input parameters
	maxs = np.max(data[:, :5], axis=0) # get maxima of 5 input parameters
	mins[1] = spin_lower_limit
	draws = np.random.uniform(mins, maxs, size=(n_draws, mins.shape[0])) # uniform draws of possible parameters for each of 5 input parameters
	files = glob.glob('trained_GPs/*')

	for file in files: # loading interpolators
		try:
			intps = np.concatenate((intps, np.array([ssg.load_gp(file+'/model')])), axis=0)
		except NameError:
			intps = np.array([ssg.load_gp(file+'/model')])

	for idx in range(draws.shape[0]):
		output = np.array([ssg.predict(intps[intp], draws[idx].reshape(1, -1)) for intp in range(intps.shape[0])]) # prediction for each param combo
		errors = output[:, 1, 0] # these are the errors for each of the output parameters
		candidate_err = np.array([np.sum(errors**2)]) # add individual output parameter errors in quadrature to get one representative error value
		try:
			all_errs = np.append(all_errs, candidate_err, axis=0)
		except NameError:
			all_errs = candidate_err
	mode_err = mode(all_errs)[0][0] # the mode will most likely be the largest 1-sigma error, i.e. where the models are performing least accurately (may change)
	idxs = np.where(all_errs>=mode_err-5e-8)[0] # 5e-8 accounts for rounding error from scipy stats' mode function sig fig cutoff
	candidates = draws[idxs]
	
	kde_limits = [[0, None], [spin_lower_limit, 1], [0, None], [0.05, 0.5], [0, None]] # mbh, sbh, mdisk, yedisk, entdisk, physical limits/bounds for KDE
	kdes, parameters = np.zeros(mins.shape[0], dtype='object'), np.zeros((n_placed, mins.shape[0]))
	kde_eval_pts = np.linspace(mins, maxs, 10000) # near-continuous representation of input parameter space
	weights = np.zeros((kde_eval_pts.shape[0], mins.shape[0])) 

	for idx in range(mins.shape[0]):
		kdes[idx] = bkde(candidates[:, idx], low=kde_limits[idx][0], high=kde_limits[idx][1]) # pdf for given parameter
		weights[:, idx] = kdes[idx].evaluate(kde_eval_pts[:, idx]) # weights are the pdf evaluated at sample points
		if idx == 3: # for the Ye, give exponential preference to lower values
			weights[:, idx] *= np.exp(-kde_eval_pts[:, idx]/(0.75*np.max(kde_eval_pts[:, idx])))
		weights[:, idx] /= np.sum(weights[:, idx]) # normalize weights to be a pdf
		parameters[:, idx] = np.random.choice(kde_eval_pts[:, idx], p=weights[:, idx], size=n_placed) # use normalized pdf as weight, re-draw from candidates
		
	param_names = ['mbh', 'sbh', 'mdisk', 'yedisk', 'entdisk']
	for idx in range(mins.shape[0]):
		samps = np.random.choice(kde_eval_pts[:, idx], p=weights[:, idx], size=kde_eval_pts.shape[0]) # verifying what the input parameter PDFs look like
		plt.hist(candidates[:, idx], bins=20, density=True)
		plt.hist(samps, bins=20, density=True)
		plt.xlim(mins[idx], maxs[idx])
		plt.savefig('plotting/placement_pdfs/%s.png' % param_names[idx])
		plt.close()

	alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761 # Foucart fitting parameters from Foucart 2018
	mass_bh, spin_bh = parameters[:, 0], parameters[:, 1]
	G, c = 6.6743e-11, 3e8 # pc km^2 M_sun^-1 s^-2, km s^-1
	mass_ns = 1.4#*1.989e30*G/(c**2) # Msun
	c_ns = np.zeros((20))
	r_ns = np.linspace(11, 15, c_ns.shape[0])*1e3 # neutron star radius in meters
	mass_disk = np.zeros((mass_bh.shape[0], c_ns.shape[0]))

	q = mass_bh/mass_ns # mass ratio m_bh / m_ns
	eta = q/(1+q)**2
	z1 = 1 + (1 - spin_bh**2)**(1/3)*((1 + spin_bh)**(1/3) + (1 - spin_bh)**(1/3))
	z2 = (3*spin_bh**2 + z1**2)**(1/2)
	r_isco_hat = 3 + z2 - np.sign(spin_bh)*((3 - z1)*(3 + z1 + 2*z2))**(1/2)
	c_ns = mass_ns*1.989e30*G/(c**2)/r_ns
	for idx in range(r_ns.shape[0]):
		mass_disk[:, idx] = np.array([alpha*((1-2*c_ns[idx])/eta**(1/3)) - beta*r_isco_hat*c_ns[idx]/eta + gamma])
		mass_disk[:, idx] = np.sign(mass_disk[:, idx]) * (np.abs(mass_disk[:, idx])**delta)

	mass_disk[np.where(mass_disk<0)] = 0 # effectively the implementation of the Max function from Eq. 4 of Foucart 2018
	#mass_disk *= (mass_ns + 0.08*mass_ns**2) # Gao, He, Ai, Shun-Ke, et al. 2019, less massive disks
	mass_disk *= mass_ns*(1 + (0.6*c_ns/(1-0.5*c_ns))) # Lattimer & Prakash 2001, more massive disks

	mass_disk_bounds = np.linspace(mins[2], maxs[2], 1000).T # uninformed bounds are what we set as the physical limits
	for idx in range(n_placed): # now we want to set our bounds to the Foucart prediction based on BH mass/spin and NS compactness
		mass_disk_bounds_samp = mass_disk_bounds[np.where((mass_disk_bounds > np.min(mass_disk[idx])) & (mass_disk_bounds < np.max(mass_disk[idx])))]
		mass_disk_weights = kdes[2].evaluate(mass_disk_bounds_samp) # evaluate the KDE at the new *limited* disk mass bounds
		mass_disk_weights /= np.sum(mass_disk_weights)
		parameters[idx, 2] = np.random.choice(mass_disk_bounds_samp, p=mass_disk_weights) # draw the disk mass value from *limited* PDF

	return parameters

def check_similar(data, parameters, rerun_threshold):
#	inputs = data[:, :5]
	reruns = np.zeros(parameters.shape[0])
	n_placed = parameters.shape[0]
	for placed in range(n_placed):
		add = True
		for existing in range(data.shape[0]):
			check = np.abs(parameters[placed]-data[existing])/np.abs(parameters[placed])
			if np.any(check < rerun_threshold):
				reruns[placed]=1
				#continue
				add = False
				break
		if add: data = np.concatenate((data, parameters[placed].reshape(1, -1)), axis=0)
	return data, reruns

n_placed = int(sys.argv[1]) # command-line argument specifying how many parameter combinations to identify (how many simulations we want to place this batch)
n_draws = 1000

files = glob.glob('outflow_properties*.dat') # collects all the outflow properties to date
for file in files:
	try:
		data = np.concatenate((data, np.loadtxt(file)), axis=0)
	except NameError:
		data = np.loadtxt(file)
data = np.delete(data, np.where(data[:, 0]==8.896), axis=0)
print('************************************************************')
print('SIZE OF TRAINING SET = ', data.shape[0])
print('************************************************************')
data = data[:, :5]
rerun_threshold = np.min([1/data.shape[0], 0.005]) # lower to accept suggested parameter values which are closer to existing parameter values
print('re-run threshold is set to: ', rerun_threshold)
parameters = placement_params(files, n_placed, n_draws=n_draws)
data, reruns = check_similar(data, parameters, rerun_threshold)
print(reruns, data.shape)
while np.any(reruns):
	parameters_redraw = placement_params(files, np.count_nonzero(reruns), n_draws=n_draws)
	#parameters[np.where(reruns!=0)] = parameters_redraw
	parameters = parameters_redraw
	data, reruns = check_similar(data, parameters, rerun_threshold)
	print(reruns, data.shape)

np.savetxt('placement_params.dat', np.c_[np.linspace(data.shape[0]-n_placed+1, data.shape[0], n_placed), data[-n_placed:]], fmt=('%d %3.5f %3.5f %3.5f %3.5f %3.5f'))

#for idx in range(n_placed):
#	print(parameters[idx])
#print(data[-n_placed:])
	
