
from import_training_data import import_training_data
from check_collinearity import check_collinearity
from reorganize_data import reorganize_data
from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import notebook
import seaborn as sns
import os.path

save_path = 'report//single_variable_visualization//'

X, Y = import_training_data()
#X, Y = data_cleaning(X, Y, 5)
X = check_collinearity(X)
X, y = reorganize_data(X, Y)
names = list(X)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

# # one variable vs another
# save_path = 'report\\variable_correlation'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# for i in range(len(names)):
#     for j in range(i+1, len(names)):
#         print names[i], names[j]
#         sns.jointplot(X[names[i]], X[names[j]], cmap=cmap, kind = 'kde')
#         plt.savefig(os.path.join(save_path, names[i] +' Vs ' + names[j]))
#         plt.close('all')


# single variable vs target (attrition)
save_path = 'report\\single_variable_visualization'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(len(names)):
    sns.jointplot(X[names[i]], y, cmap=cmap, kind = 'kde')
    plt.savefig(os.path.join(save_path, names[i]))
    plt.close('all')


# skewness of the target (attrition)
save_path = 'report\\'
if not os.path.exists(save_path):
    os.mkdir(save_path)
sns.distplot(y)
plt.savefig(os.path.join(save_path, 'distribution of target variable'))