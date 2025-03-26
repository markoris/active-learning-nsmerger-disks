import numpy as np
import glob
import save_sklearn_gp as ssg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode, rv_histogram
from sklearn.neighbors import KernelDensity
from plotutils.bounded_kde import Bounded_kde as bkde

files = glob.glob('outflow_properties*.dat')
for file in files:
	try:
		data = np.concatenate((data, np.loadtxt(file)), axis=0)
	except NameError:
		data = np.loadtxt(file)

mins = np.min(data[:, :5], axis=0)
maxs = np.max(data[:, :5], axis=0)
draws = np.random.uniform(mins, maxs, size=(10000, mins.shape[0]))

files = glob.glob('trained_GPs/*')

for file in files:
	try:
		intps = np.concatenate((intps, np.array([ssg.load_gp(file+'/model')])), axis=0)
	except NameError:
		intps = np.array([ssg.load_gp(file+'/model')])

threshold = 1e-10

for idx in range(draws.shape[0]):
	output = np.array([ssg.predict(intps[intp], draws[idx].reshape(1, -1)) for intp in range(intps.shape[0])])
	errors = output[:, 1, 0]
	candidate_err = np.array([np.sum(errors**2)])
	try:
		all_errs = np.append(all_errs, candidate_err, axis=0)
	except NameError:
		all_errs = candidate_err
mode_err = mode(all_errs)[0][0]
idxs = np.where(all_errs>=mode_err-5e-8)[0] # 5e-8 accounts for rounding error from scipy stats' mode function sig fig cutoff
candidates = draws[idxs]

#bw = candidates.shape[0]**(-1/(mins.shape[0]+4))
#
#mbh_kde = KernelDensity(kernel='gaussian', bandwidth=bw*np.std(candidates[:, 0])).fit(candidates[:, 0].reshape(-1, 1))
#sbh_kde = KernelDensity(kernel='gaussian', bandwidth=bw*np.std(candidates[:, 1])).fit(candidates[:, 1].reshape(-1, 1))
#mdisk_kde = KernelDensity(kernel='gaussian', bandwidth=bw*np.std(candidates[:, 2])).fit(candidates[:, 2].reshape(-1, 1))
#yedisk_kde = KernelDensity(kernel='gaussian', bandwidth=bw*np.std(candidates[:, 3])).fit(candidates[:, 3].reshape(-1, 1))
#entdisk_kde = KernelDensity(kernel='gaussian', bandwidth=bw*np.std(candidates[:, 4])).fit(candidates[:, 4].reshape(-1, 1))

mbh_kde = bkde(candidates[:, 0], low=0) # is there an upper limit on the BH mass? ask Jonah
sbh_kde = bkde(candidates[:, 1], low=0, high=1)
mdisk_kde = bkde(candidates[:, 2], low=0) # upper limit on disk mass? ask Jonah
yedisk_kde = bkde(candidates[:, 3], low=0, high=0.5) # 0.5 = neutral matter, upper limit of 1 would imply proton-rich, not physically realistic
entdisk_kde = bkde(candidates[:, 4], low=0) # upper limit on entropy? ask Jonah

kde_eval_pts = np.linspace(mins, maxs, 10000)

mbh = np.random.choice(kde_eval_pts[:, 0], p=mbh_kde.evaluate(kde_eval_pts[:, 0])/np.sum(mbh_kde.evaluate(kde_eval_pts[:, 0])))
sbh = np.random.choice(kde_eval_pts[:, 1], p=sbh_kde.evaluate(kde_eval_pts[:, 1])/np.sum(sbh_kde.evaluate(kde_eval_pts[:, 1])))
mdisk = np.random.choice(kde_eval_pts[:, 2], p=mdisk_kde.evaluate(kde_eval_pts[:, 2])/np.sum(mdisk_kde.evaluate(kde_eval_pts[:, 2])))
yedisk = np.random.choice(kde_eval_pts[:, 3], p=yedisk_kde.evaluate(kde_eval_pts[:, 3])/np.sum(yedisk_kde.evaluate(kde_eval_pts[:, 3])))
entdisk = np.random.choice(kde_eval_pts[:, 4], p=entdisk_kde.evaluate(kde_eval_pts[:, 4])/np.sum(entdisk_kde.evaluate(kde_eval_pts[:, 4])))

mbh_pdf = rv_histogram(np.histogram(candidates[:, 0], bins=20, density=True))
sbh_pdf = rv_histogram(np.histogram(candidates[:, 1], bins=20, density=True))
mdisk_pdf = rv_histogram(np.histogram(candidates[:, 2], bins=20, density=True))
yedisk_pdf = rv_histogram(np.histogram(candidates[:, 3], bins=20, density=True))
entdisk_pdf = rv_histogram(np.histogram(candidates[:, 4], bins=20, density=True))

print('training_set_size \t n_pdf_samples \t\t mode_of_error')
print(data.shape[0], '\t\t\t', idxs.shape[0], '\t\t\t', mode_err)
print('mass_bh \t\t spin_bh \t\t mass_disk \t\t ye_disk \t\t entropy_disk')
print(mbh_pdf.rvs(), '\t', sbh_pdf.rvs(), '\t', mdisk_pdf.rvs(), '\t', yedisk_pdf.rvs(), '\t', entdisk_pdf.rvs()) # add size=X to rvs for multiple draws
#print(mbh_kde.sample(), '\t\t', sbh_kde.sample(), '\t\t', mdisk_kde.sample(), '\t\t', yedisk_kde.sample(), '\t\t', entdisk_kde.sample()) # add n_samples=X to rvs for multiple draws
print(mbh, '\t', sbh, '\t', mdisk, '\t', yedisk, '\t', entdisk) # add n_samples=X to rvs for multiple draws
print('-----')
print('mass_bh mean:\t ', mbh_pdf.mean(), '\tmass_bh std:\t ', mbh_pdf.std())
print('spin_bh mean:\t ', sbh_pdf.mean(), '\tspin_bh std:\t ', sbh_pdf.std())
print('mass_disk mean:\t ', mdisk_pdf.mean(), '\tmass_disk std:\t ', mdisk_pdf.std())
print('ye_disk mean:\t ', yedisk_pdf.mean(), '\tye_disk std:\t ', yedisk_pdf.std())
print('ent_disk mean:\t ', entdisk_pdf.mean(), '\tent_disk std:\t ', entdisk_pdf.std())

plt.hist(candidates[:, 0], bins=20, density=True, stacked=True)
#plt.hist(mbh_kde.sample(candidates.shape[0]), bins=20, density=True, stacked=True)
plt.hist(np.random.choice(kde_eval_pts[:, 0], p=mbh_kde.evaluate(kde_eval_pts[:, 0])/np.sum(mbh_kde.evaluate(kde_eval_pts[:, 0])), size=10000), bins=20, density=True, stacked=True)
plt.xlim([mins[0], maxs[0]])
plt.title('Black Hole Mass')
plt.savefig('plotting/placement_pdfs/mbh.png')
plt.close()
plt.hist(candidates[:, 1], bins=20, density=True)
#plt.hist(sbh_kde.sample(candidates.shape[0]), bins=20, density=True, stacked=True)
plt.hist(np.random.choice(kde_eval_pts[:, 1], p=sbh_kde.evaluate(kde_eval_pts[:, 1])/np.sum(sbh_kde.evaluate(kde_eval_pts[:, 1])), size=10000), bins=20, density=True, stacked=True)
plt.xlim([mins[1], maxs[1]])
plt.title('Black Hole Spin')
plt.savefig('plotting/placement_pdfs/sbh.png')
plt.close()
plt.hist(candidates[:, 2], bins=20, density=True)
#plt.hist(mdisk_kde.sample(candidates.shape[0]), bins=20, density=True, stacked=True)
plt.hist(np.random.choice(kde_eval_pts[:, 2], p=mdisk_kde.evaluate(kde_eval_pts[:, 2])/np.sum(mdisk_kde.evaluate(kde_eval_pts[:, 2])), size=10000), bins=20, density=True, stacked=True)
plt.xlim([mins[2], maxs[2]])
plt.title('Disk Mass')
plt.savefig('plotting/placement_pdfs/mdisk.png')
plt.close()
plt.hist(candidates[:, 3], bins=20, density=True)
#plt.hist(yedisk_kde.sample(candidates.shape[0]), bins=20, density=True, stacked=True)
plt.hist(np.random.choice(kde_eval_pts[:, 3], p=yedisk_kde.evaluate(kde_eval_pts[:, 3])/np.sum(yedisk_kde.evaluate(kde_eval_pts[:, 3])), size=10000), bins=20, density=True, stacked=True)
plt.xlim([mins[3], maxs[3]])
plt.title('Disk Ye')
plt.savefig('plotting/placement_pdfs/yedisk.png')
plt.close()
plt.hist(candidates[:, 4], bins=20, density=True)
#plt.hist(entdisk_kde.sample(candidates.shape[0]), bins=20, density=True, stacked=True)
plt.hist(np.random.choice(kde_eval_pts[:, 4], p=entdisk_kde.evaluate(kde_eval_pts[:, 4])/np.sum(entdisk_kde.evaluate(kde_eval_pts[:, 4])), size=10000), bins=20, density=True, stacked=True)
plt.xlim([mins[4], maxs[4]])
plt.title('Disk Entropy')
plt.savefig('plotting/placement_pdfs/entdisk.png')
plt.close()

print('verifying drawn samples are within bounds as expected')

samps = np.random.choice(kde_eval_pts[:, 0], p=mbh_kde.evaluate(kde_eval_pts[:, 0])/np.sum(mbh_kde.evaluate(kde_eval_pts[:, 0])), size=10000)
print(np.min(samps), np.max(samps))
samps = np.random.choice(kde_eval_pts[:, 1], p=sbh_kde.evaluate(kde_eval_pts[:, 1])/np.sum(sbh_kde.evaluate(kde_eval_pts[:, 1])), size=10000)
print(np.min(samps), np.max(samps))
samps = np.random.choice(kde_eval_pts[:, 2], p=mdisk_kde.evaluate(kde_eval_pts[:, 2])/np.sum(mdisk_kde.evaluate(kde_eval_pts[:, 2])), size=10000)
print(np.min(samps), np.max(samps))
samps = np.random.choice(kde_eval_pts[:, 3], p=yedisk_kde.evaluate(kde_eval_pts[:, 3])/np.sum(yedisk_kde.evaluate(kde_eval_pts[:, 3])), size=10000)
print(np.min(samps), np.max(samps))
samps = np.random.choice(kde_eval_pts[:, 4], p=entdisk_kde.evaluate(kde_eval_pts[:, 4])/np.sum(entdisk_kde.evaluate(kde_eval_pts[:, 4])), size=10000)
print(np.min(samps), np.max(samps))
