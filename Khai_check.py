import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

def cokurt(df):
    # Number of stocks
    num = len(df.columns)
    # Tensor Product Matrix
    mtx2 = np.zeros(shape=(len(df), num ** 3))
    print(mtx2.shape)
    v = df.values
    print(v.shape)
    means = v.mean(0, keepdims=True)
    v1 = (v - means).T
    for k in range(num):
        for i in range(num):
            for j in range(num):
                vals = v1[i] * v1[j] * v1[k]
                mtx2[:, (k * (num ** 2)) + (i * num) + j] = vals / float((len(df) - 1) * df.iloc[:, i].std() * \
                                                                         df.iloc[:, j].std() * df.iloc[:, k].std())
    # cokurtosis matrix
    m4 = np.dot(v1, mtx2)
    for i in range(num ** 3):
        use = i % num
        m4[:, i] = m4[:, i] / float(df.iloc[:, use].std())
    return m4

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

