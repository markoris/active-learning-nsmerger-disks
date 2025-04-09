import numpy as np

# 2 params to fit avg: alpha, beta
AIC_avg = 2*2 - 2*(-0.04048852997419902)

peak_lnL_mbh_mdisk = np.array([
-0.0009401712008371171,
-2.1236165473406368e-07,
-7.062863567420353e-05,
-2.3174564965212913e-05,
-0.0002537987672204954,
-0.0005763716913305905,
-0.007274835714108269,
-0.010172219477584704,
-0.0018471295854868488,
-0.0016198061402537321,
-0.0008165452204037602,
-0.003863101101266283,
-0.002770415869888854,
-0.0017870992319984898,
-0.00037377313199988437,
-0.0026574270550769248,
-0.00017166055962185734,
-0.00021180103737248804,
-0.0010425045594975794,
-1.4434878030184858e-06,
-0.0005612780517362698,
-0.00021852000640194293,
-0.00025309123779987446,
-0.00015266351831530852,
-0.0006392897373339789])

# 5 params to fit per disk: A, B, C, D, E for alpha = A*Mbh + B*Mdisk + C, beta = D*alpha + E
AIC_per_disk = 2*5 - 2*peak_lnL_mbh_mdisk

print('Preferred model is the one with the minimum AIC value')
print('criterion = AIC_avg - AIC_per_disk')
print('criterion > 0 prefers per-disk fits, criterion < 0 prefers average fit')
print('lnL avg: ', -0.04048852997419902)
print('lnL per disk: ', peak_lnL_mbh_mdisk)
print('AIC avg: ', AIC_avg)
print('AIC per disk: ', AIC_per_disk)
print('criterion: ', AIC_avg - AIC_per_disk)
