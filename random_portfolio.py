import pandas as pd
import numpy as np
from scipy.optimize import minimize
# kur_matrix = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/kur_matrix.csv')
# print(kur_matrix)
from main import coskew, cokurt, ef, calculateSharpe, max_drawdown
from Portfolio_Momentum import max_sharpe, max_sharpe_bounded_norml2, max_sharpe_bounded_norml2_kurtosis, dict_weight
train_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/train_data.csv')
test_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/test_data.csv')

kur_matrix = cokurt(train_data)
max_sharpe_sol1 = max_sharpe(dataframe= train_data, upperbound= 0.1)
x = max_sharpe_sol1
kurtosis_sharpe = x.T.dot(kur_matrix).dot(np.kron(np.kron(x, x), x))
dist_sharpe = np.sqrt(x.T.dot(x))

delta = 0.8
norml2_sol1 = max_sharpe_bounded_norml2(dataframe= train_data, upperbound= 0.1, delta= delta, dist= dist_sharpe)
combine_sol1 = max_sharpe_bounded_norml2_kurtosis(dataframe = train_data, upperbound = 0.1, delta = delta, kurtosis = kurtosis_sharpe
                                                 , kurtosis_matrix = kur_matrix, dist = dist_sharpe )



col = train_data.columns

print('sharpe solution',dict_weight(max_sharpe_sol1,col))
print('norml2 solution',dict_weight(norml2_sol1,col))
print('combine solution',dict_weight(combine_sol1,col))

max_sharpe_return = np.array(test_data*10**9).dot(max_sharpe_sol1.T)
norml2_return = np.array(test_data*10**9).dot(norml2_sol1.T)
combine_return = np.array(test_data*10**9).dot(combine_sol1.T)


print(f'Normal sharpe stop {calculateSharpe(max_sharpe_return)}')
print(f'norml2 sharpe stop {calculateSharpe(norml2_return)}')
print(f'combine sharpe stop {calculateSharpe(combine_return)}')

print(f'dd sharpe stop {max_drawdown(10**9,max_sharpe_return)}')
print(f'dd norml2 stop {max_drawdown(10**9,norml2_return)}')
print(f'dd combine stop {max_drawdown(10**9,combine_return)}')
print('STOPPPPPPPPPP')
