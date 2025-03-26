n_placed=8
n_backups=2
module purge
module load gcc/7.4.0 openmpi/2.1.2 hdf5-parallel/1.8.16 python/3.8-anaconda-2020.07
echo 'loaded necessary modules'
cd active-learning-nsmerger-disks
python gp_fit.py
echo 'finished training GPs'
python gp_placement.py $(($n_placed+$n_backups))
echo 'finished determining suggested simulation parameters'
#cat placement_params.dat >> /users/mristic/jonah_sims/placement/parameter_history.dat
cd ..
python fishbone_moncrief_params.py active-learning-nsmerger-disks/placement_params.dat
awk '{print $1 " " $2 " " $3 " " $4 " " $5}' torus_parameters.dat >> parameter_history.dat
cd /users/mristic/jonah_sims/nubhlight/prob/torus_cbc/
while IFS=" " read -r mbh spin mdisk ye ent mdmeasured rin rmax rhounit munit; do
	cd /users/mristic/jonah_sims/nubhlight/prob/torus_cbc/
	filename="mbh$(printf "%.3f" $mbh)_a$(printf "%.3f" $spin)_mdisk$(printf "%.3f" $mdisk)_ye$(printf "%.3f" $ye)_s$(printf "%.3f" $ent)"
	echo $filename
	#python build.py -dir /net/scratch4/mristic/disk_simulations/$filename -nu -2d -hdf -M ${mbh} -a ${spin} -ent ${ent} -Ye ${ye} -rin ${rin} -rmax ${rmax} -rho ${rhounit}
	python build.py -dir /net/scratch4/mristic/disk_simulations/$filename -nu -3d -hdf -M ${mbh} -a ${spin} -ent ${ent} -Ye ${ye} -rin ${rin} -rmax ${rmax} -rho ${rhounit}
	cp /users/mristic/jonah_sims/utils/torus.badger.sbatch /net/scratch4/mristic/disk_simulations/$filename
	sed -i "s/#SBATCH -J a0p9375/#SBATCH -J $filename/" /net/scratch4/mristic/disk_simulations/$filename/torus.badger.sbatch
	cd /net/scratch4/mristic/disk_simulations/$filename
done < /users/mristic/jonah_sims/placement/torus_parameters.dat
cd /users/mristic/jonah_sims/placement/
sh execution.sh
