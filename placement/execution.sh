for d in /net/scratch4/mristic/disk_simulations/*/;
do
	cd $d
	echo "Starting run for $d at"
	date
	sbatch torus.badger.sbatch
	cd ..
done

while [ $(ls /net/scratch4/mristic/disk_simulations | wc -l) -gt 0 ];
do
	while [ $(squeue -u mristic --local | wc -l) -gt 1 ]; do
		sleep 3600
	done
	for d in /net/scratch4/mristic/disk_simulations/*; do
		cd $d
		all_slurm_err_files="$d/slurm*.err"
		latest_slurm_err_file=( $all_slurm_err_files )
		if grep -q "DUE TO TIME LIMIT\|CANCELLED AT" ${latest_slurm_err_file[-1]}; then
			echo "Resuming run for $d at"
			date
			sbatch torus.badger.sbatch
		else
			mv $d/ /net/scratch4/mristic/for_processing/
			echo "Moved finished run $d to /net/scratch4/for_processing/"
		fi
	done
done

cd /users/mristic/jonah_sims/placement/

sh post_process.sh
