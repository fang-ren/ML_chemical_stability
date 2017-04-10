

from import_training_data import import_training_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

save_path = 'report//'

def check_collinearity(X):
    # get column names from the variables
    names = list(X)

    # check collinearity of variables
    corr = X.corr()

    # # visualize correlation
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.matshow(abs(corr))
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90, fontsize = 5)
    # plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 5)
    # plt.savefig(save_path + 'correlation', dpi = 600)

    correlated = []
    for i in range(corr.shape[0]-1):
        for j in range(i+1, corr.shape[1]):
            if abs(corr.ix[i][j]) > 0.7:
                correlated.append(names[j])
                if names[j] in X:
                    del X[names[j]]

    # visualize correlation after delete dependent variable
    # corr = X.corr()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.matshow(abs(corr))
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90, fontsize = 5)
    # plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 5)
    # plt.savefig(save_path + 'correlation_checked', dpi = 600)


    return X

# X, Y = import_training_data()
# X = check_collinearity(X)

#print data, data.shape

