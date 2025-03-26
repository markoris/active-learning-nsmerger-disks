import sys, glob, h5py
sys.path.append('/users/mristic/jonah_sims/nubhlight/script/analysis/')
from hdf5_to_dict import load_geom, load_hdr, TracerData
from plot_tracers import plot_minor_summary, get_theta, get_mass_mdot, get_vr
import numpy as np
from natsort import natsorted
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

print('mbh, abh, mdisk0, ye0, s0, <vr>, <ye>, <T>, <rho>, tmax')
for sim in sim_dirs:
    sim_params = sim.split('/')[-1]
    #print(sim_params)
    sim_params = sim_params.split('_')
    #print(sim_params)
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
    #print(mbh, abh)
    if skip_conditions.any(): 
        continue
    
    if (mbh == 2.58 and abh == 0.690): 
        print("2.58 & 0.69 & 0.12 & 0.10 & 4.00 & 0.07 & 0.25 & 2.65 & 5.417e+05 & 127 \\\\")
        continue

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

    # time in seconds
    time = np.copy(tracer.data['time'])
    time *= tracer.units['T_unit'] # convert time from geometrical units to seconds

    rho = np.copy(tracer.data['rho'])
    rho *= tracer.units['RHO_unit']

    # mdot + mass through surface at r=250M or ~1000km
    # mass is cumulative mass through dA at angle theta
    theta_range, t_range, mdot, mass = get_mass_mdot(tracer, nbins_theta=3, nbins_time=10)
    mdot, mass = mdot*(tracer.units['M_unit']/tracer.units['T_unit']/M_sol_cgs), mass*tracer.units['M_unit']/M_sol_cgs
    # convert mdot from geometrical units per second, to grams/sec, to solar masses per second
    # convert mass from geometrical units, to grams, to solar masses
    # time given in seconds

    # temperature in GK
    temperature = np.copy(tracer.data['T'])
    temperature *= GK_per_MeV # convert from MeV to GK

    # Ye
    ye = np.copy(tracer.data['Ye'])

    # mass-weights (not a probability distribution, just pure mass-weight)
    # can be normalized to a probability distribution if needed
    mass_weights = np.copy(tracer['mass'])
    mass_weights *= tracer.units['M_unit']/M_sol_cgs # convert from geometrical units, to grams, and then to solar mass

    # angle in radians
    angle = get_theta(tracer)
    
    # radial velocity
    vr = get_vr(tracer, geom)
    vr *= tracer.units['L_unit']/tracer.units['T_unit']/c_cgs # convert from geometrical units, to cm/s, and then to fraction of speed of light

    plt.hist(vr, bins=25)
    
    plt.savefig('figures/{}.pdf'.format(str(mbh)+'_'+str(abh)+'_'+str(mdisk_init)))
    plt.close()
    
    vr_mass_avg = np.average(vr, weights=mass_weights)
    ye_mass_avg = np.average(ye, weights=mass_weights)
    temp_mass_avg = np.average(temperature, weights=mass_weights)
    rho_mass_avg = np.average(rho, weights=mass_weights)
    max_time = time.max()
    
    print('{0:.2f} & {1:.2f} & {2:.3g} & {3:.2f} & {4:.2f} & {5:.3f} & {6:.2f} & {7:4.2f} & {8:.3e} & {9:3.0f} & {10:.3g} & {11:.3g} \\\\'.format(mbh, abh, mdisk_init, ye_init, s_init, vr_mass_avg, ye_mass_avg, temp_mass_avg, rho_mass_avg, max_time*1000, mass[:, -1].sum(), np.std(vr)))

#    print("times in seconds :", time)
#    print('-----------------------------')
#
#    print("max density in g/cm^3 :", rho.max())
#    print('-----------------------------')
#
#    print("mdot in msol across %d angle bins spanning " % len(theta_range), "%d-%d" %(theta_range.min(), theta_range.max()), " degrees and %d time bins spanning " % len(t_range), "%g-%g" % (t_range.min(), t_range.max()), " seconds: ",  mdot)
#    print('-----------------------------')
#
#    print("mass in msol across %d angle bins spanning " % len(theta_range), "%d-%d" %(theta_range.min(), theta_range.max()), " degrees and %d time bins spanning " % len(t_range), "%g-%g" % (t_range.min(), t_range.max()), " seconds: ",  mass)
#    print('-----------------------------')
#  
#    # sum over mass angles, not mass times because mass is cumulative 
#    print('total mass in msol :', mass.sum(axis=0))
#    print('-----------------------------')
#    
#    print('percent of ejected mass to initial disk mass :', mass.sum()/mdisk_init*100)
#    print('-----------------------------')
# 
#    print("temperature in GK :", temperature)
#    print('-----------------------------')
#
#    print("Ye :", ye)
#    print('-----------------------------')
#
#    print("mass weights :", mass_weights)
#    print('-----------------------------')
#
#    print("angle in radians :", angle)
#    print('-----------------------------')
#
#    # testing filter as a function of time to get "snapshots"
#    # make windows at least 1 ms wide
#    mask = np.logical_and(time > 300e-3, time < 320e-3) # get tracers between 300 and 320 ms
#    print("n_tracers before filter: ", len(tracer))
#    print("n_tracers after 300-320 ms filter: ", len(tracer.filter(mask)))
#    print('-----------------------------')
#
#    print("radial velocity as fraction of c :", vr)
#
#    #np.savetxt('parameter_summary.dat', np.c_[time, temperature, ye, 
#    
#    #plot_minor_summary(tracer, geom, savename=sim.split('/')[-1]+'.pdf')
#
