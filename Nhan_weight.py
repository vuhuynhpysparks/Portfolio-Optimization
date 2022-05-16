import pandas as pd
import numpy as np
from scipy.optimize import minimize
# kur_matrix = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/kur_matrix.csv')
# print(kur_matrix)
from main import coskew, cokurt, ef, calculateSharpe, max_drawdown
from Portfolio_Momentum import max_sharpe, max_sharpe_bounded_norml2, max_sharpe_bounded_norml2_kurtosis, dict_weight, max_sharpe_kurtosis, min_kurtosis
# train_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/train_data.csv')
full_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/full_data.csv', index_col= 0, parse_dates= [0])
# full_data['datetime'] = pd.to_datetime(full_data['datetime'])
# full_data['week_num'] = full_data['datetime'].dt.week
# full_data.set_index('datetime', inplace = True)
print(full_data)

def processing_v1(data_input, delta  = 0.9):
    kur_matrix = cokurt(data_input)
    max_sharpe_sol = max_sharpe(dataframe=data_input, upperbound=0.2)
    x = max_sharpe_sol
    kurtosis_sharpe = x.T.dot(kur_matrix).dot(np.kron(np.kron(x, x), x))
    dist_sharpe = np.sqrt(x.T.dot(x))

    combine_sol = max_sharpe_bounded_norml2_kurtosis(dataframe = data_input, upperbound = 0.2, delta = delta, kurtosis = kurtosis_sharpe
                                                     , kurtosis_matrix = kur_matrix, dist = dist_sharpe)
    return combine_sol


def processing_v2(data_input, delta  = 0.9):
    kur_matrix = cokurt(data_input)
    min_kur_solution, min_kur = min_kurtosis(upperbound= 0.2, cokurtosis_matrix=kur_matrix)
    max_sharpe_sol = max_sharpe_kurtosis(dataframe= data_input, delta = delta, upperbound=0.2, kurtosis_matrix= kur_matrix,kurtosis= min_kur )
    x = max_sharpe_sol
    # kurtosis_sharpe = x.T.dot(kur_matrix).dot(np.kron(np.kron(x, x), x))
    dist_sharpe = np.sqrt(x.T.dot(x))
    combine_sol = max_sharpe_bounded_norml2_kurtosis_v2(dataframe=data_input, upperbound=0.2, delta=delta,
                                                                 kurtosis=min_kur
                                                                 , kurtosis_matrix=kur_matrix, dist=dist_sharpe)
    return combine_sol




def max_sharpe_bounded_norml2_kurtosis_v2( dataframe, upperbound, delta, kurtosis, kurtosis_matrix, dist):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1},
            {'type': 'ineq', 'fun': lambda x: -sum(x[i] ** 2 for i in range(n)) + (delta*dist)**2},
            {'type': 'ineq', 'fun': lambda x: - x.T.dot(kurtosis_matrix).dot(np.kron(np.kron(x, x), x)) + (1/delta) * kurtosis}]
    bnds = [(0, upperbound)] * n
    bnds = tuple(bnds)
    cons = tuple(cons)
    inital_point = np.array([1 / n] * n)
    sol = minimize(f, x0=inital_point, method='SLSQP', bounds=bnds, constraints=cons)
    if sol.success == False:
        print('NOT EXISTS OPTIMIZED WEIGHT')
    else:
        print('EXISTS OPTIMIZED WEIGHT')
    return sol.x


from datetime import datetime
from_time_rotate_weight= '2021-06-01'
data_dict = []
index_dt = []
for dt in pd.bdate_range(from_time_rotate_weight, datetime.now(), freq='1W'):
    # weight = 1
    weight = processing_v2(full_data[full_data.index <= dt])
    weightt = {key.split('.')[0]: value for key, value in (dict_weight(weight,full_data.columns)).items()}
    data_dict.append(weightt)
    index_dt.append(dt)

df_weight = pd.DataFrame(data_dict, index=index_dt)
df_weight.to_csv('/home/vu/Desktop/Portfolio-Optimization/weight_v2.csv')





# col = full_data.columns
#
# print('sharpe solution',dict_weight(max_sharpe_sol1,col))
# print('norml2 solution',dict_weight(norml2_sol1,col))
# print('combine solution',dict_weight(combine_sol1,col))
#
# max_sharpe_return = np.array(test_data*10**9).dot(max_sharpe_sol1.T)
# norml2_return = np.array(test_data*10**9).dot(norml2_sol1.T)
# combine_return = np.array(test_data*10**9).dot(combine_sol1.T)
#
#
# print(f'Normal sharpe stop {calculateSharpe(max_sharpe_return)}')
# print(f'norml2 sharpe stop {calculateSharpe(norml2_return)}')
# print(f'combine sharpe stop {calculateSharpe(combine_return)}')
#
# print(f'dd sharpe stop {max_drawdown(10**9,max_sharpe_return)}')
# print(f'dd norml2 stop {max_drawdown(10**9,norml2_return)}')
# print(f'dd combine stop {max_drawdown(10**9,combine_return)}')
# print('STOPPPPPPPPPP')
