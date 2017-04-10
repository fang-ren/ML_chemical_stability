# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
import imp

import_training_data = imp.load_source("import_test_data", "import_test_data.py")
save_path = 'report//'

def reorganize_test_data(test_data):
    X_training = test_data

    # mock-up data
    # X_training = np.array([[1,2,3],[4,5,6]])
    # Y_training = np.array([[0,1,0,1,0,1,0,1],[1,0,0,0,0,0,0,1]])

    # print X_training
    # print Y_training

    samples = X_training.shape[0]
    features = X_training.shape[1]
    layer = 9

    samples_new = samples * layer
    features_new = features +1

    X_training_new = [[]]* samples_new

    for i in range(samples):
        for j in range(layer):
            X_training_new[i*layer+j] = np.concatenate((X_training[i], [j]))

    test_data_new = np.array(X_training_new)

    return test_data_new

## to run
# training_data = import_training_data.import_training_data()
# X_training_new, Y_training_new = reorganize_training_data(training_data)