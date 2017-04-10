# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
import pandas as pd
from import_training_data_new import import_training_data
from check_collinearity import check_collinearity

save_path = 'report//'

def reorganize_data(X, Y):
    sample_num = X.shape[0]
    feature_num = X.shape[1]
    layer = Y.shape[1]

    # create re-organized dataframe
    X_new = pd.DataFrame()
    Y_new = pd.DataFrame()

    # create new feature: composition_A
    composition_A = []

    # the organized Y vector will only have 1 column
    stability = []

    for i in range(layer):
        X_new = X_new.append(X, ignore_index=True)
        composition_A += [i/10.0]*sample_num
        stability += list(Y.ix[:,i])

    X_new['composition_A'] = composition_A
    Y_new['stability'] = stability
    return X_new, Y_new

# ## to run
# X, Y = import_training_data()
# X = check_collinearity(X)
# X, Y = reorganize_data(X, Y)