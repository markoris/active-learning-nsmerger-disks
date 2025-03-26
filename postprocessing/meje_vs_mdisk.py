import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def line(x, a, b):
    return a*x + b

plt.rc('font', size=20)

data = np.genfromtxt('summary.dat')
meje = np.log10(data[:, 20])
mdisk = np.log10(data[:, 4])

popt, pcov = curve_fit(line, mdisk, meje)
perr = np.sqrt(np.diag(pcov))
ul = popt + perr
ll = popt - perr
mdisks = np.log10(np.logspace(mdisk.min(), mdisk.max(), 100))
ul = line(mdisks, *ul)
ll = line(mdisks, *ll)
print(np.linalg.cond(pcov))
plt.figure(figsize=(8, 6))
plt.scatter(mdisk, meje, color='black')
plt.plot(mdisk, line(mdisk, *popt), 'r-')
#plt.fill_between(mdisk, line(mdisk, *(popt-perr)), line(mdisk, *(popt+perr)), color='red', alpha=0.3)
plt.fill_between(mdisks, ll, ul, color='red', alpha=0.5)
plt.text(0.55, 0.45, r'$\log_{10} M_{\rm{eje}} = a \log_{10} M_{\rm{disk}} + b$', transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.4, r'$a = {0:.4g} \pm {1:.3g}$'.format(popt[0], perr[0]), transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.35, r'$b = {0:.4g} \pm {1:.3g}$'.format(popt[1], perr[1]), transform=plt.gca().transAxes, size=14)
plt.xlabel(r'$M_{\rm{disk}}$')
plt.ylabel(r'$M_{\rm{eje}}$')
plt.tight_layout()
plt.savefig('../figures/meje_vs_mdisk.pdf')

