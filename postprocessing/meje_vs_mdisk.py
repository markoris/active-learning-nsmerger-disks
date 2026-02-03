import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def line(x, a, b):
    return a*x + b

def line_CI(x, pcov):
    return x**2*pcov[0, 0] + pcov[1, 1] + 2*x*pcov[0, 1]

plt.rc('font', size=20)

data = np.genfromtxt('../paper_data/summary.dat', usecols=np.linspace(0, 20, 11).astype(int), skip_header=1)
meje = data[:, 10]
mdisk = data[:, 2]
meje, mdisk = np.delete(meje, 20), np.delete(mdisk, 20)

for i, md in enumerate(mdisk):
    print(i, md)

print(meje/mdisk)

meje = np.log10(meje)
mdisk = np.log10(mdisk)

popt, pcov = curve_fit(line, mdisk, meje)

#perr = np.sqrt(np.diag(pcov))
#ul = popt + perr
#ll = popt - perr
#print(popt)
#print(ll, ul)
mdisks = np.log10(np.logspace(mdisk.min(), mdisk.max(), 100))
mej_fit = line(mdisks, *popt)
ul = mej_fit + np.sqrt(line_CI(mdisks, pcov))
ll = mej_fit - np.sqrt(line_CI(mdisks, pcov))
print(np.linalg.cond(pcov))

## Monte Carlo fit
#
#draws = np.random.uniform(low=[0, -3], high=[3, 0], size=(int(1e7), 2))
#for md in mdisk:
#    try:
#        preds = np.concatenate((preds, line(md, *draws.T)[None, :]), axis=0)
#    except NameError:
#        preds = line(md, *draws.T)[None, :]
#residuals = np.sum((preds.T - meje)**2, axis=1)
#mu, sigma = np.mean(residuals), np.std(residuals)
#res_gaussian = 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-(residuals - mu)**2 / (2 * sigma**2))
#sort_idxs = np.argsort(res_gaussian)
#res_gaussian = res_gaussian[sort_idxs]
#draws = draws[sort_idxs] 
#lower_lim = np.percentile(res_gaussian, 50-34.1) # 1-sigma
#fit = np.percentile(res_gaussian, 50)
#upper_lim = np.percentile(res_gaussian, 50+34.1) # 1-sigma
#idx_lower = np.argmin(np.abs(res_gaussian - lower_lim))
#idx_fit = np.argmin(np.abs(res_gaussian - fit))
#idx_upper = np.argmin(np.abs(res_gaussian - upper_lim))
#popt2 = draws[idx_fit]
#ll2, ul2 = draws[idx_lower], draws[idx_upper]
#print(popt2)
#print(ll2, ul2)

#mej_fit = line(mdisks, *popt2)
#ul = line(mdisks, *ul2)
#ll = line(mdisks, *ll2)

plt.figure(figsize=(8, 6))
plt.scatter(mdisk, meje, color='black')
plt.plot(mdisks, mej_fit, 'r-')
plt.fill_between(mdisks, ul, ll, color='red', alpha=0.5)
plt.text(0.55, 0.45, r'$\log_{10} M_{\rm{eje}} = a \log_{10} M_{\rm{disk}} + b$', transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.39, r'$a = {0:.4g} \pm {1:.3g}$'.format(popt[0], pcov[0, 0]), transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.33, r'$b = {0:.4g} \pm {1:.3g}$'.format(popt[1], pcov[1, 1]), transform=plt.gca().transAxes, size=14)
plt.xlabel(r'$M_{\rm{disk}}$')
plt.ylabel(r'$M_{\rm{eje}}$')
plt.tight_layout()
plt.savefig('../figures/meje_vs_mdisk.pdf')

