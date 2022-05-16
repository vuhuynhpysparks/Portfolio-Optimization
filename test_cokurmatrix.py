import pymarkowitz.Moments as mm
import pandas as pd
train_data = pd.read_csv('/home/vu/Desktop/Portfolio-Optimization/alpha_data/HSG_clayton_gng_7.csv')
print(train_data)
moment_generate = mm.MomentGenerator(train_data)
co_kur_mat = moment_generate.calc_cokurt_mat()
print(co_kur_mat)