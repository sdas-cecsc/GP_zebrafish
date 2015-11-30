import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('rmse_info_analysis.csv', dtype=float, delimiter=',')

x = data[:,0]
y_lds_min = data[:,1]
y_lds_max = data[:,2]
y_lds_mean = data[:,3]
y_hds_min = data[:,4]
y_hds_max = data[:,5]
y_hds_mean = data[:,6]

plt.plot(x, y_lds_min, 'r', label='min rmse_lds')
plt.plot(x, y_lds_max, 'g', label='max rmse_lds')
plt.plot(x, y_lds_mean, 'b', label='mean rmse_lds')
plt.plot(x, y_hds_min, 'c', label='min rmse_hds')
plt.plot(x, y_hds_max, 'm', label='max rmse_hds')
plt.plot(x, y_hds_mean, 'y', label='mean rmse_hds')

plt.xlabel('#dependencies')
plt.ylabel('amplitude')
plt.title('variation acc. to dependencies')

legend = plt.legend(loc='upper right')

plt.show()