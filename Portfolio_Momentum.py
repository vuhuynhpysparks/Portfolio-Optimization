import pandas as pd
import numpy as np
from scipy.optimize import minimize
# kur_matrix = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/kur_matrix.csv')
# print(kur_matrix)
from main import coskew, cokurt, ef, calculateSharpe, max_drawdown
train_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/train_data.csv')
test_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/test_data.csv')
# kur_matrix = cokurt(train_data)

# upperbound = 0.2
# kur_matrix = np.array(kur_matrix)
# print(kur_matrix.shape)
# f = lambda x: x.T.dot(kur_matrix).dot(np.kron(np.kron(x, x), x))
# num_asset = kur_matrix.shape[0]
# cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(num_asset)) - 1}]
# bnds = [(0, upperbound)] * num_asset
# bnds = tuple(bnds)
# cons = tuple(cons)
# initial_point = np.array([1 / num_asset] * num_asset)
# # initial_point  = np.array([0] * num_asset)
# # initial_point[3] = 1/2
# # initial_point[4] = 1/2
# sol = minimize(f, x0= initial_point, method='SLSQP', bounds=bnds, constraints= cons)
#
# if sol.success == False:
#     print('NOT EXISTS OPTIMIZED WEIGHT')
# else:
#     print('EXISTS OPTIMIZED WEIGHT')
#     print(sol.x)
# kurtosis = sol.fun

def min_kurtosis(upperbound, cokurtosis_matrix):
    f = lambda x: x.T.dot(cokurtosis_matrix).dot(np.kron(np.kron(x, x), x))
    num_asset = cokurtosis_matrix.shape[0]
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(num_asset)) - 1}]
    bnds = [(0, upperbound)] * num_asset
    bnds = tuple(bnds)
    cons = tuple(cons)
    initial_point = np.array([1 / num_asset] * num_asset)
    # initial_point  = np.array([0] * num_asset)
    # initial_point[3] = 1/2
    # initial_point[4] = 1/2
    sol = minimize(f, x0=initial_point, method='SLSQP', bounds=bnds, constraints=cons)
    if sol.success == False:
        print('NOT EXISTS OPTIMIZED WEIGHT')
    else:
        print('EXISTS OPTIMIZED WEIGHT')
        print(sol.x)
    return sol, sol.fun
    # kurtosis = sol.fun


def max_sharpe(dataframe, upperbound):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1}]
    bnds = [(0, upperbound)] * n
    bnds = tuple(bnds)
    cons = tuple(cons)
    inital_point = np.array([1 / n] * n)
    sol = minimize(f, x0=inital_point, method='SLSQP', bounds=bnds, constraints= cons)
    if sol.success == False:
        print('NOT EXISTS OPTIMIZED WEIGHT')
    else:
        print('EXISTS OPTIMIZED WEIGHT')
    return sol.x


def max_sharpe_kurtosis( dataframe, upperbound, delta, kurtosis, kurtosis_matrix):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1},
            {'type': 'ineq', 'fun': lambda x: - x.T.dot(kurtosis_matrix).dot(np.kron(np.kron(x, x), x)) + 1/delta * kurtosis}]
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

def max_sharpe_bounded_norml2( dataframe, upperbound, delta, dist):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1},
            {'type': 'ineq', 'fun': lambda x: -sum(x[i] ** 2 for i in range(n)) + (delta*dist)**2}]
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

def max_sharpe_bounded_norml2_kurtosis( dataframe, upperbound, delta, kurtosis, kurtosis_matrix, dist):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1},
            {'type': 'ineq', 'fun': lambda x: -sum(x[i] ** 2 for i in range(n)) + (delta*dist)**2},
            {'type': 'ineq', 'fun': lambda x: - x.T.dot(kurtosis_matrix).dot(np.kron(np.kron(x, x), x)) + delta * kurtosis}]
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


def dict_weight(vector_weight,column):
    dict = {}
    for i in range(len(column)):
        if abs(vector_weight[i]) < 1e-4:
            dict[column[i]] = 0
        else:
            dict[column[i]] = vector_weight[i]
    return dict
# max_sharpe_sol = max_sharpe(dataframe= train_data, upperbound= 0.2)
#
# dist_sharpe = np.sqrt(max_sharpe_sol.T.dot(max_sharpe_sol))
#
# delta = 0.95
#
# norml2_sol = max_sharpe_bounded_norml2(dataframe= train_data, upperbound= 0.2, delta= delta, dist= dist_sharpe)
#
#
# #kustosis_processing
# kurtosis_sol = max_sharpe_kurtosis(dataframe= train_data, upperbound= 0.2, delta= delta,
#                                                  kurtosis = kurtosis, kurtosis_matrix= kur_matrix)
#
# dist_kurtosis = np.sqrt(kurtosis_sol.T.dot(kurtosis_sol))
# combine_sol = max_sharpe_bounded_norml2_kurtosis(dataframe= train_data, upperbound= 0.2, delta= delta,
#                                                  kurtosis = kurtosis, kurtosis_matrix= kur_matrix, dist= dist_kurtosis)
#
#
#
# col = train_data.columns
#
# print('sharpe solution',dict_weight(max_sharpe_sol,col))
# print('norml2 solution',dict_weight(norml2_sol,col))
# print('combine solution',dict_weight(combine_sol,col))
#
# max_sharpe_return = np.array(test_data*10**9).dot(max_sharpe_sol.T)
# norml2_return = np.array(test_data*10**9).dot(norml2_sol.T)
# combine_return = np.array(test_data*10**9).dot(combine_sol.T)
#
#
#
# print(f'Normal sharpe {calculateSharpe(max_sharpe_return)}')
# print(f'norml2 sharpe {calculateSharpe(norml2_return)}')
# print(f'combine sharpe {calculateSharpe(combine_return)}')
#
# print(f'dd sharpe {max_drawdown(10**9,max_sharpe_return)}')
# print(f'dd norml2 {max_drawdown(10**9,norml2_return)}')
# print(f'dd combine {max_drawdown(10**9,combine_return)}')




