#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH -N 1
#SBATCH -n 75
#SBATCH --cpus-per-task=1
#SBATCH --hint=multithread
#SBATCH --account=t25_nltekn
#SBATCH --job-name=missing_tracers
#SBATCH --output=slurm-%J.out

module load python PrgEnv-cray cray-hdf5-parallel

for d in /lustre/scratch4/turquoise/mristic/disk_sims/mbh*;
do
#    if [[ $d == "/lustre/scratch4/turquoise/mristic/disk_sims/mbh2.580_a0.690_mdisk0.120_ye0.100_s4.000" ]]; then
#        continue
#    fi
	count=`ls -1 $d/*.td 2>/dev/null | wc -l`
	if [ $count != 0 ]; then
		continue
	fi
	cd $d
	echo "Accumulating and pruning tracers for $(basename $d)"
	python3 -u /users/mristic/jonah_sims/nubhlight/script/analysis/accumulate_and_prune_tracers.py $d/dumps/tracers/ tracers_accumulated_r250.td 250 $d/dumps/tracers/pruned/ tracers_accumulated_r250_nse.td 5 -n 10
done
