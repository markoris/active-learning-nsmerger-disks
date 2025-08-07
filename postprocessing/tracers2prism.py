import os, sys, glob, h5py, gc, subprocess, shutil
sys.path.append('/users/mristic/astro/jonah_sims/nubhlight/script/analysis/')
from hdf5_to_dict import load_geom, load_hdr, TracerData
from plot_tracers import plot_minor_summary, get_theta, get_mass_mdot, get_vr
from tracers_data_to_traces import make_if_needed, save_traces_in_dir
import tracers_to_PRISM
import numpy as np
from natsort import natsorted
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# point to simulations copied over from archive

sim_dirs = np.array(glob.glob('/lustre/scratch5/mristic/astro/disk_sims/*'))
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
    simname = np.copy(sim_params)
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

    Ye = np.copy(tracer.data['Ye'])
    theta = 90-np.degrees(get_theta(tracer))
    Ye_bins = np.linspace(0, 0.5, 11)
    theta_bins = np.degrees([np.arccos(1 - 2*(i-1)/54) for i in range(1, 56)])
    mass_weights = np.copy(tracer['mass'])
    mass_weights *= tracer.units['M_unit']/M_sol_cgs # convert from geometrical units, to grams, and then to solar mass
    n = 0
    keys = ['id', 'Ye', 'theta']
    selected_tracers = {k: [] for k in keys}
    #for i in range(len(Ye_bins)-1):
    #    lower_bound_Ye = round(Ye_bins[i], 2)
    #    upper_bound_Ye = round(Ye_bins[i+1], 2)
        #print(lower_bound, upper_bound, tracer.filter(mask).data['id'].shape[0])
    #    mask_Ye = np.logical_and(Ye > lower_bound_Ye, Ye < upper_bound_Ye)
    for j in range(len(theta_bins)-1):
        lower_bound_theta = theta_bins[j]
        upper_bound_theta = theta_bins[j+1]
        mask_theta = np.logical_and(theta > lower_bound_theta, theta < upper_bound_theta)
    #    mask = np.logical_and(mask_Ye, mask_theta)
        mask = mask_theta
        tracer_masked = tracer.filter(mask)
        mass_weights_masked = tracer_masked['mass'] * tracer_masked.units['M_unit']/M_sol_cgs
        n += tracer_masked.data['id'].shape[0]
        if len(tracer_masked.data['id']) == 0: continue
        selected_id = np.random.choice(tracer_masked.data['id'], p=mass_weights_masked/mass_weights_masked.sum(), size=1)[0]
        selected_tracers['id'].append(selected_id)
        selected_tracers['Ye'].append(tracer_masked.filter(tracer_masked.data['id']==selected_id).data['Ye'][0])
        selected_tracers['theta'].append(upper_bound_theta)
    for k in selected_tracers.keys():
        selected_tracers[k] = np.array(selected_tracers[k])

    tracer_trimmed = TracerData.fromfile(sim+'/tracers_accumulated_r250.td', ids=selected_tracers['id'])

    tids = selected_tracers['id']
    nts = len(tids)
    print("There are {} traces to save".format(nts))
  
    try: shutil.rmtree(f'./traces/{simname}')
    except FileNotFoundError: pass
    make_if_needed(f'traces/{simname}') 
 
    print("Saving...")
    i = 0
    print("...in directory {}...".format(f'traces/{simname}'))
    gc.collect()
    i = save_traces_in_dir(i,nts,f'traces/{simname}',tids,tracer_trimmed)

    pwd = os.getcwd()

    try: shutil.rmtree(f'./prism_inputs/{simname}')
    except FileNotFoundError: pass
    subprocess.call(['python', '/users/mristic/astro/jonah_sims/nubhlight/script/analysis/tracers_to_PRISM.py', f'{pwd}/traces/{simname}', f'{pwd}/prism_inputs/{simname}', '-f', '/lustre/scratch5/mristic/codes/prism/nse_calc', '--single'])

    orig_fnames = ['trace_{:08d}'.format(int(tid)) for tid in selected_tracers['id']]

    subprocess.call(['head', '-30', f'prism_inputs/{simname}/{orig_fnames[13]}/trajectory.dat'])

    for i in range(len(orig_fnames)):
        print(orig_fnames[i])
        print(selected_tracers['theta'][i])
        os.rename(f'prism_inputs/{simname}/{orig_fnames[i]}', f'prism_inputs/{simname}/{selected_tracers["theta"][i]}')

    subprocess.call(['head', '-30', f'prism_inputs/{simname}/{selected_tracers["theta"][13]}/trajectory.dat'])
    sys.exit()
