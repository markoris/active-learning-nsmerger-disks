module purge
module load python/3.8-anaconda-2020.07 gcc/7.4.0 openmpi/2.1.2 hdf5-parallel/1.8.16
for d in /net/scratch4/mristic/for_processing/mbh*;
do
	cd $d
	macc=$(python /users/mristic/jonah_sims/nubhlight/script/analysis/get_mass_accreted.py "$d/dumps/")
	macc="$(cut -d':' -f2 <<<"$macc")"
	echo "Accumulating and pruning tracers for $(basename $d)"
	python /users/mristic/jonah_sims/nubhlight/script/analysis/accumulate_and_prune_tracers.py $d/dumps/tracers/ tracers_accumulated_r250.td 250 $d/dumps/tracers/pruned/ tracers_accumulated_r250_nse.td 5 -n 10
	#python /users/mristic/jonah_sims/nubhlight/script/analysis/accumulate_and_prune_tracers.py $d/dumps/tracers/ tracers_accumulated_r100.td 100 $d/dumps/tracers/pruned/ tracers_accumulated_r100_nse.td 5 -n 10 &> accumulate_and_prune_tracers.out # 100 for 2D sim, use 250 for 3D sims
	sim_inputs=($(python /users/mristic/jonah_sims/placement/sim_inputs.py $(basename $d)))
	#python /users/mristic/jonah_sims/nubhlight/script/analysis/get_mass_Ye_from_tracers.py --mbh ${sim_inputs[0]} -a ${sim_inputs[1]} --mdisk ${sim_inputs[2]} --ye ${sim_inputs[3]} -s ${sim_inputs[4]} --macc $macc --tracers tracers_accumulated_r100.td --nse tracers_accumulated_r100_nse.td > outflow_properties.dat	
	python /users/mristic/jonah_sims/nubhlight/script/analysis/get_mass_Ye_from_tracers.py --mbh ${sim_inputs[0]} -a ${sim_inputs[1]} --mdisk ${sim_inputs[2]} --ye ${sim_inputs[3]} -s ${sim_inputs[4]} --macc $macc --tracers tracers_accumulated_r250.td --nse tracers_accumulated_r250_nse.td > outflow_properties.dat	
	sed -n '20p' outflow_properties.dat >> /users/mristic/jonah_sims/placement/active-learning-nsmerger-disks/outflow_properties_active_learning.dat
	echo "$(basename $d) simulation ready for archiving" >> /net/scratch4/mristic/for_processing/email.txt
done
mv /net/scratch4/mristic/for_processing/* /net/scratch4/mristic/for_archiving/
echo "Sending email to notify that simulations are ready for archiving"
mail -s "simulations ready for archiving" mristic@lanl.gov < /net/scratch4/mristic/for_archiving/email.txt
