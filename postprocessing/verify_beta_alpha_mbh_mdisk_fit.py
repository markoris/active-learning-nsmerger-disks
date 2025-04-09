import sys, glob, h5py
sys.path.append('/users/mristic/jonah_sims/nubhlight/script/analysis/')
from hdf5_to_dict import load_geom, load_hdr, TracerData
from plot_tracers import plot_minor_summary, get_theta, get_mass_mdot, get_vr
import numpy as np
from natsort import natsorted
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import beta, rv_histogram, norm
import warnings
from itertools import islice

with open('../paper_data/vr_mbh_mdisk_fit_params.dat', 'r') as f:
    beta_fit_params = np.loadtxt(islice(f, 3))
    alpha_fit_params = np.loadtxt(f)
f.close()

def beta_param(alpha):
    a, b, da, db = beta_fit_params
    return a*alpha[0] + b, (a-da)*alpha[1] + (b-db), (a+da)*alpha[2] + (b+db)

def alpha_param(mbh, mdisk):
    a, b, c, da, db, dc = alpha_fit_params
    return a*mbh + b*np.log10(mdisk) + c, (a-da)*mbh + (b-db)*np.log10(mdisk) + (c-dc), (a+da)*mbh + (b+db)*np.log10(mdisk) + (c+dc)

# point to simulations copied over from archive

sim_dirs = np.array(glob.glob('/lustre/scratch4/turquoise/mristic/disk_sims/*'))
sim_dirs = natsorted(sim_dirs)

# physical constants

M_sol_cgs = 1.989e33 # solar mass in grams
c_cgs = 2.998e10 # speed of light in cm/s
sigma_boltzmann = 8.617333262e-5 # eV/K
sigma_boltzmann /= 1e6 # MeV/K
sigma_boltzmann *= 1e9 # MeV/GK
GK_per_MeV = 1/sigma_boltzmann # Boltzmann constant in GK/MeV

# not all of these will have tracer_accumulated_rXXX.td files
# that's fine, I can always re-accumuluate tracers at some radius
# and re-run the summary script on those. for now, set up for what exists

for sim in sim_dirs:
    sim_params = sim.split('/')[-1]
    sim_params = sim_params.split('_')
    mbh = float(sim_params[0][3:])
    abh = float(sim_params[1][1:])
    mdisk_init = float(sim_params[2][5:])
    ye_init = float(sim_params[3][2:])
    s_init = float(sim_params[4][1:])

    skip_conditions = np.array([ (mbh == 2.31), 
                        (mbh == 2.58 and abh == 0.938),
                        (mbh == 2.67 and abh == 0.690),
                        (mbh == 3.072),
                        (mbh == 4.430),
                      ])
    if skip_conditions.any(): 
        continue
   
    # GW170817 disk giga sim, post-process later, for now auto-fill from previous summary table 
    if (mbh == 2.58 and abh == 0.690): 
        #print("2.58 & 0.69 & 0.12 & 0.10 & 4.00 & 0.07 & 0.25 & 2.65 & 5.417e+05 & 127 \\\\")
        continue

    print(sim_params)

    try:
        tracer = TracerData.fromfile(sim+'/tracers_accumulated_r250.td')
    except FileNotFoundError:
        print("No tracer available for ", sim.split('/')[-1])
        continue

    try:
        hdr = load_hdr(sim+'/dumps/dump_00000000.h5')
    except FileNotFoundError:
        try:
            hdr = load_hdr(sim+'/dumps/dump2d_00000005.h5')
        except FileNotFoundError:
            print('No dump file available for ', sim.split('/')[-1])

    geom = load_geom(hdr)
    
    ### QUANTITIES OF INTEREST ###
    
    # example code from plot summary
    # the log-normal distribution is used to set the 2d histogram
    # color-scale to a log-scale so that larger mass tracers don't
    # overshadow the lower mass bins 
    #### norm = LogNorm(vmin=10.**(-7.5),vmax=10.**(-4),clip=True)

    # the weights are calculated based on the mass of the tracer
    # normalized by the total mass of the system to create a probability distribution
    # weights = tracers['mass']*units['M_unit']/cgs['MSOLAR']

    # mass-weights (not a probability distribution, just pure mass-weight)
    # can be normalized to a probability distribution if needed
    mass_weights = np.copy(tracer['mass'])
    mass_weights *= tracer.units['M_unit']/M_sol_cgs # convert from geometrical units, to grams, and then to solar mass

    # angle in radians
    angle = get_theta(tracer)
   
    # radial velocity
    vr = get_vr(tracer, geom)
    vr *= tracer.units['L_unit']/tracer.units['T_unit']/c_cgs # convert from geometrical units, to cm/s, and then to fraction of speed of light

    a = alpha_param(mbh, mdisk_init)
    b = beta_param(a)

    def func(args):
        a, b = args
        return -np.sum(mass_weights*np.log10(beta.pdf(vr, a, b, loc=0)))

    x = np.linspace(beta.ppf(0.01, a[0], b[0]), beta.ppf(0.99, a[0], b[0]), 100)
    plt.plot(x, beta.pdf(x, a[0], b[0]), 'r-')
    #plt.fill_between(x, beta.pdf(x, a[1], b[1]), beta.pdf(x, a[2], b[2]), color='r', alpha=0.3)
    plt.hist(vr, bins=25, density=True, weights=mass_weights)
    plt.savefig('../figures/vr_fit_to_mbh_mdisk/vr_mbh_mdisk{}.pdf'.format(str(mbh)+'_'+str(abh)+'_'+str(mdisk_init)))
    plt.close()
    print(func((a[0], b[0])))
