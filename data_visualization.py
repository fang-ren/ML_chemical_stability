
from import_training_data import import_training_data
from check_collinearity import check_collinearity
from reorganize_data import reorganize_data
from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt

save_path = 'report//single_variable_visualization//'

X, Y = import_training_data()
#X, Y = data_cleaning(X, Y, 5)
X = check_collinearity(X)
X, Y = reorganize_data(X, Y)


names = list(X)

for i in range(len(names)):
    plt.plot(X[names[i]], Y, 'o', markersize = 2)
    plt.grid()
    plt.savefig(save_path + names[i] + 'png')

    plt.close('all')