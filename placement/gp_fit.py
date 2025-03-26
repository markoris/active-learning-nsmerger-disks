import os
import glob
import save_sklearn_gp as ssg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# Columns:
# [0]:  Mass of black hole (solar masses)
# [1]:  Spin of black hole (unitless)
# [2]:  Disk mass at the initial time (solar masses)
# [3]:  Disk Ye at the initial time (unitless)
# [4]:  Disk entropy at initial time (k_b/baryon)
# [5]:  Total mass in outflow (solar masses)
# [6]:  Ratio of total mass in outflow to mass accreted
# [7]:  Mass-averaged Ye in total outflow (unitless)
# [8]:  Extrapolated total mass in outflow to late times
# [9]:  Mass in polar region, >= 50 degress off equator (solar masses)
# [10]: Ratio of mass in polar region to mass accreted
# [11]: Mass-averaged Ye in polar region
# [12]: Extrapolated mass in polar region
# [13]: Mass in equatorial region, <= 15 degrees off equator (solar masses)
# [14]: Ratio of mass in equatorial region to mass accreted
# [15]: Mass-averaged Ye in equatorial region
# [16]: Extrapolated mass in equatorial region
# Note that polar + equatorial != total!

cols_short = {0: "massbh",                                                                                                                                    1: "spinbh",                                                                                                                                                  2: "initdiskmass",                                                                                                                                            3: "initdiskye",                                                                                                                                              4: "initdiskentropy",                                                                                                                                         5: "totalmassoutflow",                                                                                                                                        6: "ratiomasstotalaccreted",                                                                                                                                  7: "massavgyetotal",                                                                                                                                          8: "masspolar",                                                                                                                                               9: "masspolar",                                                                                                                                               10: "ratiomasspolaraccreted",                                                                                                                                 11: "massavgyepolar",                                                                                                                                         12: "masspolarextrapol",                                                                                                                                      13: "massequator",                                                                                                                                            14: "ratiomassequatoraccreted",                                                                                                                               15: "massavgyeequator",                                                                                                                                       16: "massequatorextrapol",}  

files = glob.glob('outflow_properties*.dat')
for file in files:
	try:
		data = np.concatenate((data, np.loadtxt(file)), axis=0)
	except NameError:
		data = np.loadtxt(file)

#print(np.where(data[:, 0] == 8.896))
#data = np.delete(data, np.where(data[:, 0]==8.896), axis=0)

x_train = data[:, :5]
y_train = data[:, 5:]

errors = 0

for parameter in range(y_train.shape[1]):
	kernel = WhiteKernel() + C()*RBF() # add more parameters to kernels, such as from slick, once training set gets larger
	gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=True)	
	gpr.fit(x_train, y_train[:, parameter])
	try:
		os.mkdir('trained_GPs/%s' % cols_short[parameter+x_train.shape[1]])
	except FileExistsError: pass
	ssg.export_gp_compact('trained_GPs/%s/model' % cols_short[parameter+x_train.shape[1]], gpr)
