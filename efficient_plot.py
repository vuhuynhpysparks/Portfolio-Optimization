import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import cokurt, coskew, ef
train_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/train_data.csv')
print(train_data)
log_ret = train_data

m4 = cokurt(log_ret)

m3 = coskew(log_ret)

vol_arr, ret_arr, cokurt_arr, coskew_arr = ef(log_ret, m3, m4)

#Plot EF with skew cmap
plt.figure()
plt.scatter(vol_arr, ret_arr, c=coskew_arr, cmap='winter')
plt.colorbar(label='Skew')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()

#Plot EF with Kurt cmap
plt.figure()
plt.scatter(vol_arr, ret_arr, c=cokurt_arr, cmap='winter')
plt.colorbar(label='Kurtosis')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()
