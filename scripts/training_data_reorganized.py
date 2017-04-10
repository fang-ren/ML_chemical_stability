# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
from import_training_data_new import impor
save_path = 'report//'

def reorganize_training_data(training_data):
    X_training = training_data[:, :96]
    Y_training = training_data[:, 96:]

    # mock-up data
    # X_training = np.array([[1,2,3],[4,5,6]])
    # Y_training = np.array([[0,1,0,1,0,1,0,1],[1,0,0,0,0,0,0,1]])

    # print X_training
    # print Y_training

    samples = X_training.shape[0]
    features = X_training.shape[1]
    layer = Y_training.shape[1]

    samples_new = samples * layer
    features_new = features +1

    X_training_new = [[]]* samples_new
    Y_training_new = [[]]* samples_new

    for i in range(samples):
        for j in range(layer):
            X_training_new[i*layer+j] = np.concatenate((X_training[i], [j]))
            Y_training_new[i*layer+j] = Y_training[i][j]

    X_training_new = np.array(X_training_new)
    Y_training_new = np.array(Y_training_new)

    return X_training_new, Y_training_new

## to run
X, Y = import_training_data()
# X_training_new, Y_training_new = reorganize_training_data(training_data)