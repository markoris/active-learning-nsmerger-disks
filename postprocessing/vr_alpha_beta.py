import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

try: os.remove('../paper_data/vr_mbh_mdisk_fit_params.dat')
except FileNotFoundError: pass

# mbh, mdisk
data = np.genfromtxt('../paper_data/vr_beta_fit_parameters.dat')
mbh = data[:, 0]
mdisk = data[:, 4]
alphas = data[:, 10]
betas = data[:, 12] 
print(mbh.shape, mdisk.shape, alphas.shape, betas.shape)

mdisk = np.log10(mdisk)

def line(x, a, b):
    return a*x + b

def line2d(X, a, b, c):
    x, y = X
    return a*x + b*y + c

plt.rc('font', size=20)

popt, pcov = curve_fit(line, alphas, betas)
perr = np.sqrt(np.diag(pcov))
ul = popt + perr
ll = popt - perr
alphas_fit = np.linspace(alphas.min(), alphas.max(), 100)
ul = line(alphas_fit, *ul)
ll = line(alphas_fit, *ll)
print(np.linalg.cond(pcov))
plt.figure(figsize=(8, 6))
plt.scatter(alphas, betas, color='black')
plt.plot(alphas, line(alphas, *popt), 'r-')
plt.fill_between(alphas_fit, ll, ul, color='red', alpha=0.5)
plt.text(0.55, 0.45, r'$\beta = a \alpha + b$', transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.4, r'$a = {0:.4g} \pm {1:.3g}$'.format(popt[0], perr[0]), transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.35, r'$b = {0:.4g} \pm {1:.3g}$'.format(popt[1], perr[1]), transform=plt.gca().transAxes, size=14)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.tight_layout()
plt.savefig('../figures/vr_alphas_betas.pdf')
plt.close()

with open('../paper_data/vr_mbh_mdisk_fit_params.dat', 'a') as f:
    f.write('# beta = a * alpha(mbh, mdisk) + b (note: different a, b than above!) \n')
    f.write('# a \t b \t delta_a \t delta_b \n')
    f.write('{0:.4g} \t {1:.4g} \t {2:.4g} \t {3:.4g} \n'.format( \
            popt[0], popt[1], perr[0], perr[1]))
    f.write('\n')
f.close()

popt, pcov = curve_fit(line2d, (mbh, mdisk), alphas)
perr = np.sqrt(np.diag(pcov))
ul = popt + perr
ll = popt - perr
mbh_fit = np.linspace(mbh.min(), mbh.max(), 100)
mdisk_fit = np.log10(np.logspace(mdisk.min(), mdisk.max(), 100))
ul = line2d((mbh_fit, mdisk_fit), *ul)
ll = line2d((mbh_fit, mdisk_fit), *ll)
print(np.linalg.cond(pcov))
plt.figure(figsize=(8, 6))
#plt.scatter(mbh, mdisk, c=alphas)
plt.scatter(mbh, mdisk, c=np.abs([alphas-line2d((mbh, mdisk), *popt)]))
plt.colorbar(label='abs(true-fit)')
#plt.fill_between(mdisk_fit, ll, ul, color='red', alpha=0.5)
plt.text(0.55, 0.45, r'$\alpha = a M_{\rm{bh}} + b M_{\rm{disk}} + c$', transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.4, r'$a = {0:.4g} \pm {1:.3g}$'.format(popt[0], perr[0]), transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.35, r'$b = {0:.4g} \pm {1:.3g}$'.format(popt[1], perr[1]), transform=plt.gca().transAxes, size=14)
plt.text(0.55, 0.30, r'$c = {0:.4g} \pm {1:.3g}$'.format(popt[2], perr[2]), transform=plt.gca().transAxes, size=14)
plt.xlabel(r'$M_{\rm{bh}}$')
plt.ylabel(r'$M_{\rm{disk}}$')
plt.tight_layout()
plt.savefig('../figures/vr_mbh_mdisk_alphas.pdf')
plt.close()

with open('../paper_data/vr_mbh_mdisk_fit_params.dat', 'a') as f:
    f.write('# alpha = a*m_bh + b*m_disk + c \n')
    f.write('# a \t b \t c \t delta_a \t delta_b \t delta_c \n')
    f.write('{0:.4g} \t {1:.4g} \t {2:.4g} \t {3:.4g} \t {4:.4g} \t {5:.4g} \n'.format( \
            popt[0], popt[1], popt[2], perr[0], perr[1], perr[2]))
f.close()    

plt.scatter(alphas, betas, c=mdisk)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.colorbar()
plt.savefig('../figures/alpha_beta_mdisk.pdf')
plt.close()

plt.scatter(alphas, betas, c=mbh)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.colorbar()
plt.savefig('../figures/alpha_beta_mbh.pdf')
plt.close()
