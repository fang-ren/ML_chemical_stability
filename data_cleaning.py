
from import_training_data_new import import_training_data
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def data_cleaning(X, Y, z_score):
    # only keep the data within zscore sigma
    X_cleaned = X[(np.abs(stats.zscore(X)) < z_score).all(axis=1)]
    Y_cleaned = Y[(np.abs(stats.zscore(X)) < z_score).all(axis=1)]
    return X_cleaned, Y_cleaned



# save_path = 'report//data_cleaning_3sigma//'
#
# X, Y = import_training_data()
# names = list(X)
# feature_num = X.shape[1]
# for i in range(feature_num):
#     column = X[names[i]]
#     plt.subplot(211)
#     right = np.max(column)
#     left = np.min(column)
#     plt.hist(column,bins = np.linspace(left,right,100))
#
#     column_new = column[np.abs(stats.zscore(column)) < 3]
#     plt.subplot(212)
#     plt.hist(column_new,bins = np.linspace(left,right,100))
#     plt.savefig(save_path + names[i] + '.png')
#     plt.close('all')
#
