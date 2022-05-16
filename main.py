
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coskew(df):
    # Number of stocks
    num = len(df.columns)
    # Two dimionsal matrix for tensor product
    mtx = np.zeros(shape=(len(df), num ** 2))
    v = df.values
    means = v.mean(0, keepdims=True)
    v1 = (v - means).T
    for i in range(num):
        for j in range(num):
            vals = v1[i] * v1[j]
            mtx[:, (i * num) + j] = vals / float((len(df) - 1) * df.iloc[:, i].std() * df.iloc[:, j].std())
    # coskewness matrix
    m3 = np.dot(v1, mtx)

    # Normalize by dividing by standard deviation
    for i in range(num ** 2):
        use = i % num
        m3[:, i] = m3[:, i] / float(df.iloc[:, use].std())
    return m3


def cokurt(df):
    # Number of stocks
    num = len(df.columns)
    # Tensor Product Matrix
    mtx2 = np.zeros(shape=(len(df), num ** 3))
    v = df.values
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


def ef(df, m3, m4):
    np.random.seed(10)
    trials = 10000
    all_weights = np.zeros((trials, df.shape[1]))

    ret_arr = np.zeros(trials)
    vol_arr = np.zeros(trials)
    coskew_arr = np.zeros(trials)
    cokurt_arr = np.zeros(trials)

    for i in range(trials):
        # Get random weights and store in weight array
        weights = np.array(np.random.random(df.shape[1]))
        weights = weights / np.sum(weights)
        all_weights[i, :] = weights

        # Expected return
        ret_arr[i] = np.sum((df.mean() * weights * 252))

        # Volatility
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))

        # Coskew
        coskew_arr[i] = np.dot(weights.T, np.dot(m3, np.kron(weights, weights)))

        # Cokurtosis
        cokurt_arr[i] = np.dot(weights.T, np.dot(m4, np.kron(weights, np.kron(weights, weights))))

    fig = plt.figure()
    ax = plt.gca(projection='3d')
    a = ax.scatter(vol_arr, cokurt_arr, ret_arr, c= coskew_arr, cmap='winter')
    a = fig.colorbar(a, label='Skew')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Kurtosis')
    ax.set_zlabel('Return')
    plt.show()
    return vol_arr, ret_arr, cokurt_arr, coskew_arr

def calculateSharpe(npArray):
    sr = npArray.mean() / npArray.std() * np.sqrt(252)
    return sr

def max_drawdown(booksize, returnSeries):
    mdd = 0
    a = np.cumsum(returnSeries)
    X = a + booksize
    peak = X[0]
    dds = []
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / booksize
        if dd > mdd:
            mdd = dd
            dds.append(X[X == x])
    # print("MDD AT ", dds[-1].index[0] if len(dds) else None)
    return mdd