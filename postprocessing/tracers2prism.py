import os, sys, glob, h5py, gc, subprocess, shutil
#sys.path.append('/users/mristic/codes/nubhlight/script/analysis/')
sys.path.insert(0, '/lustre/scratch5/mristic/codes/nubhlight/script/analysis')
from hdf5_to_dict import load_geom, load_hdr, TracerData
from plot_tracers import plot_minor_summary, get_theta, get_mass_mdot, get_vr, get_bernoulli
from tracers_data_to_traces import make_if_needed, save_traces_in_dir, get_subdirs
import units; cgs = units.get_cgs()
import numpy as np
from natsort import natsorted
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

print("If too many files open error, do 'ulimit -n 100000'")

# physical constants

M_sol_cgs = 1.989e33 # solar mass in grams
c_cgs = 2.998e10 # speed of light in cm/s
sigma_boltzmann = 8.617333262e-5 # eV/K
sigma_boltzmann /= 1e6 # MeV/K
sigma_boltzmann *= 1e9 # MeV/GK
GK_per_MeV = 1/sigma_boltzmann # Boltzmann constant in GK/MeV

prism_T9 = 10.1

def draw_tracer(tracers_masked, dumps_array):

    # get the mass weights for the tracers in the current angular bin
    mass_weights_masked = tracers_masked['mass'] * tracers_masked.units['M_unit']/M_sol_cgs

    # random mass-weighted draw of ONE tracer in current angular bin
    selected_id = np.random.choice(tracers_masked.data['id'], p=mass_weights_masked/mass_weights_masked.sum(), size=1)[0]
    
    mask = np.isin(dumps_array[0]['Step#0']['id'][:], selected_id) # in the h5part[i/2001]['Step#0']['id'] list of tracer ids, is our tracer there?
    tracer_T9_init = dumps_array[0]['Step#0']['T'][mask][0]*cgs['MEV']/cgs['GK'] # if yes, get its temperature!

    # delete removes entries where mask = True
    good_tracers = np.copy(tracers_masked.data['id'])
    while tracer_T9_init < prism_T9:
        #print(len(good_tracers))
        mask = np.isin(good_tracers, selected_id) 
        good_tracers = np.delete(good_tracers, mask)
        mass_weights_masked = np.delete(mass_weights_masked, mask)
        #print(len(good_tracers))
        selected_id = np.random.choice(good_tracers, p=mass_weights_masked/mass_weights_masked.sum(), size=1)[0]
        mask = np.isin(dumps_array[0]['Step#0']['id'][:], selected_id) # in the h5part[i/2001]['Step#0']['id'] list of tracer ids, is our tracer there?
        tracer_T9_init = dumps_array[0]['Step#0']['T'][mask][0]*cgs['MEV']/cgs['GK'] # if yes, get its temperature!

    return selected_id

np.random.seed(10)

# point to simulations copied over from archive

sim_dirs = np.array(glob.glob('/lustre/scratch5/mristic/astro/disk_sims/*'))
sim_dirs = natsorted(sim_dirs)

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
                        (mbh == 8.896)
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

    #geom = load_geom(hdr)
    Ye = np.copy(tracer.data['Ye'])
    Ye_bins = np.linspace(0, 0.5, 15)

    # create directory where the 54 traces (one per angle bin) will be stored 
    try: shutil.rmtree(f'./traces/{simname}')
    except FileNotFoundError: pass
    make_if_needed(f'traces/{simname}') 

    #low_vr_tids = []
    #high_vr_tids = []
    tids = []

    pruned_tracers = natsorted(glob.glob(f'{sim}/dumps/tracers/pruned/*'))
    dumps_array = {}
    for i in range(len(pruned_tracers)):
        dumps_array[i] = h5py.File(pruned_tracers[i], 'r')

    Ye_bin_masses = []

    # loop over angular bins
    for j in range(len(Ye_bins)-1):

        # identify lower and upper bounds of each angular bin
        lower_bound_Ye = Ye_bins[j]
        upper_bound_Ye = Ye_bins[j+1]
        #print(lower_bound_Ye)
        #print(upper_bound_Ye)

        # generate and apply tracer mask based on current angle bin
        mask_Ye = np.logical_and(Ye > lower_bound_Ye, Ye < upper_bound_Ye)
        tracer_masked = tracer.filter(mask_Ye)
        
        bernoulli_parameter = get_bernoulli(tracer_masked)
        #print(bernoulli_parameter)
        mask_bernoulli = bernoulli_parameter > 0
        tracer_masked = tracer_masked.filter(mask_bernoulli)

        print(len(tracer_masked))

        if len(tracer_masked) == 0: 
            Ye_bin_masses.append(0)
            continue
        else:
            try:
                selected_id = draw_tracer(tracer_masked, dumps_array)
                tids.append(selected_id)
                Ye_bin_masses.append((tracer_masked['mass'] * tracer_masked.units['M_unit']/M_sol_cgs).sum())
            except ValueError:
                # no tracers hot enough
                print('--- No tracers above 10 GK ---')
                Ye_bin_masses.append(0)
                continue

        ###selected_id = draw_tracer(tracer_masked, dumps_array)

        ###tids.append(selected_id)

        #vr = get_vr(tracer_masked, geom)
        #vr *= tracer.units['L_unit']/tracer.units['T_unit']/c_cgs # convert from geometrical units, to cm/s, and then to fraction of speed of light

        #mask_low_vr = vr < 0.08
        #mask_high_vr = vr >= 0.08

        #tracer_low_vr = tracer_masked.filter(mask_low_vr)
        #tracer_high_vr = tracer_masked.filter(mask_high_vr)

        ##bernoulli_parameter = get_bernoulli(tracer_masked)
        ##print(bernoulli_parameter)
        ##mask_bernoulli = bernoulli_parameter > 0
        ##tracer_masked = tracer_masked.filter(mask_bernoulli)
    

        ## overcomplicating before
        ## it's simple: draw a tracer, check it's 0th (initial) dump file temp, ensure it's above 10 GK
        ## if not, redraw until true
        ## TODO: check with Jonah if we need to check whether tracer is in last dump file (i.e. perhaps not extracted?)

        ## here we put the extraction radius temperature into temps
        #temps = np.copy(tracer_masked['T']*cgs['MEV']/cgs['GK'])

        #if len(tracer_low_vr) == 0:
        #    low_vr_housekeeping = -np.inf
        #else:
        #    low_vr_id = draw_tracer(tracer_low_vr, dumps_array)
        #    low_vr_tids.append(low_vr_id)
        #    low_vr_housekeeping = 
        #high_vr_id = draw_tracer(tracer_high_vr, dumps_array)

        #high_vr_tids.append(high_vr_id)

        #print(Ye_bins[j], low_vr_id, high_vr_id)

        #### Let's ignore the vr split for now -- some Ye bins have a low or high velocity component, but not the other
        #### Adjusting for these statistical discrepancies seems annoying given the number of disks we have
        #### In Lund+ (2025), they only had the one disk, so it was easier (and each Ye bin had a low/high vr tracer)
        #### Can revisit later if we feel strongly about this

        #break

    mask = np.isin(tracer.data['id'], tids) # allegedly, tracer IDs...
    tracer_masked = tracer.filter(mask)
    tracer_masked.save('Ye_tracers.td')
    
    #mask = np.isin(tracer.data['id'], low_vr_tids) # allegedly, tracer IDs...
    #tracer_masked = tracer.filter(mask)
    #tracer_masked.save('low_vr_Ye_tracers.td')
    #
    #mask = np.isin(tracer.data['id'], high_vr_tids) # allegedly, tracer IDs...
    #tracer_masked = tracer.filter(mask)
    #tracer_masked.save('high_vr_Ye_tracers.td')
    
    pwd = os.getcwd()
    
    subprocess.call(['python', '/lustre/scratch5/mristic/codes/nubhlight/script/analysis/tracers_data_to_traces.py', '-i', 'Ye_tracers.td', '-d',  f'{pwd}/traces/{simname}', f'{sim}/dumps/tracers/pruned', '-n', '1'])
 
    #subprocess.call(['python', '/lustre/scratch5/mristic/codes/nubhlight/script/analysis/tracers_data_to_traces.py', '-i', 'low_vr_Ye_tracers.td', '-d',  f'{pwd}/traces/{simname}', f'{sim}/dumps/tracers/pruned', '-n', '1'])
    #
    #subprocess.call(['python', '/lustre/scratch5/mristic/codes/nubhlight/script/analysis/tracers_data_to_traces.py', '-i', 'high_vr_Ye_tracers.td', '-d',  f'{pwd}/traces/{simname}', f'{sim}/dumps/tracers/pruned', '-n', '1'])
    
    # purge prism_inputs directory if ran before
    try: shutil.rmtree(f'./prism_inputs/{simname}')
    except FileNotFoundError: pass
   
    make_if_needed(f'./prism_inputs/{simname}')
 
    # call tracers_to_PRISM.py script using .td files in /traces/{simname},
    # outputting to /prism_iputs/{simname}, 
    # using the --single flag to load tracers by ID rather than by time
    # and the --serial flag since running in single ID mode
    subprocess.call(['python', '/lustre/scratch5/mristic/codes/nubhlight/script/analysis/tracers_to_PRISM.py', f'{pwd}/traces/{simname}', f'{pwd}/prism_inputs/{simname}', '--single', '--serial', '-T9', f'{prism_T9}', '-a', '0.1'])

    orig_fnames = ['trace_{:08d}'.format(int(tid)) for tid in tids]
    #low_vr_orig_fnames = ['trace_{:08d}'.format(int(tid)) for tid in low_vr_tids]
    #high_vr_orig_fnames = ['trace_{:08d}'.format(int(tid)) for tid in high_vr_tids]

    Ye_idx = np.array(Ye_bin_masses).astype(bool)

    for i in range(len(orig_fnames)):
        os.rename(f'{pwd}/prism_inputs/{simname}/{orig_fnames[i]}', f'{pwd}/prism_inputs/{simname}/{Ye_bins[:-1][Ye_idx][i]}')
        # in here, list through orig_fnames_lowvr, orig_fnames_highvr, put them in a shared Ye_bins parent directory, then rename
        #make_if_needed(f'{pwd}/prism_inputs/{simname}/{Ye_bins[i]}')
        #os.rename(f'{pwd}/prism_inputs/{simname}/{low_vr_orig_fnames[i]}/trajectory.dat', f'{pwd}/prism_inputs/{simname}/{Ye_bins[i]}/trajectory_low_vr.dat')
        #os.rename(f'{pwd}/prism_inputs/{simname}/{high_vr_orig_fnames[i]}/trajectory.dat', f'{pwd}/prism_inputs/{simname}/{Ye_bins[i]}/trajectory_high_vr.dat')
        #shutil.rmtree(f'{pwd}/prism_inputs/{low_vr_orig_fnames[i]}')
        #shutil.rmtree(f'{pwd}/prism_inputs/{high_vr_orig_fnames[i]}')

    np.savetxt(f'{pwd}/prism_inputs/{simname}/Ye_bin_mass.dat', np.c_[Ye_bins[:-1], Ye_bin_masses], header=f'Total outflow mass = {np.sum(Ye_bin_masses)}')

