# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import imp
from sklearn.metrics import f1_score

import_training_data = imp.load_source("import_training_data", "import_training_data.py")
import_test_data = imp.load_source("import_test_data", "import_test_data.py")
reorganize_training_data = imp.load_source("reorganize_training_data", "training_data_reorganized.py")
save_path = 'report//'

training_data = import_training_data.import_training_data()

# to prune training_data
training_data = np.concatenate((training_data[:, :96], training_data[:,97:-1]), axis = 1)
X_training = training_data[:, :96]
Y_training = training_data[:, 96:]


#
# ################################
# # original training data format
#
# # split training data into two parts: training_sub and training_validate to assess the performance
# X_training_sub = X_training[:2000]
# Y_training_sub = Y_training[:2000]
#
# X_validate = X_training[2000:]
# Y_validate = Y_training[2000:]
#
# # neural network training
# hidden_layer_configs = [(48,), (96, 48, 1), (96, 96, 48, 24, 1), (96, 96, 48, 48, 24, 24, 1), (96, 96, 48, 48, 24, 24, 12, 12, 1)]
#
# f_measures1 = []
# for hidden_layer in hidden_layer_configs:
#     Y_predict1 = []
#     clf = [[] for i in range(Y_training.shape[1])]
#     for i in range(Y_training.shape[1]):
#         # clf[i] = MLPClassifier(hidden_layer_sizes= hidden_layer, activation = 'logistic', solver='lbfgs')
#         clf[i] = MLPClassifier(hidden_layer_sizes=hidden_layer, solver='lbfgs')
#         # clf[i] = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='logistic')
#         # clf[i] = MLPClassifier(hidden_layer_sizes=hidden_layer)
#         # print clf[i]
#         clf[i].fit(X_training_sub, Y_training_sub[:,i])
#         Y_predict = clf[i].predict(X_validate)
#         Y_predict1 = np.concatenate((Y_predict1, Y_predict))
#     f_measure1 = f1_score(Y_validate.reshape(5148,), Y_predict1)
#     print Y_predict1
#     print f_measure1
#     f_measures1.append(f_measure1)
#
#
# Y_predict1 = []
# for i in range(Y_training.shape[1]):
#     clf = MLPClassifier(hidden_layer_sizes=hidden_layer_configs[2], solver = 'lbfgs')
#     clf.fit(X_training_sub, Y_training_sub[:, i])
#     Y_predict = clf.predict(X_validate)
#     Y_predict1 = np.concatenate((Y_predict1, Y_predict))
# Y_predict1 = Y_predict1.reshape(9, 572).T
#
# # visualize Y_predict and Y_validate(truth)
# plt.figure(1, (18, 6))
# plt.subplot(131)
# plt.plot([1, 3, 5, 7, 9], f_measures1, 'o')
# plt.ylim(0, 1)
# plt.xlabel('# of hidden layers')
# plt.ylabel('f-measure')
#
# plt.subplot(132)
# plt.title('prediction')
# x_axis = [i + 1 for i in range(Y_predict1.shape[1] + 1)]
# y_axis = [i + 1 for i in range(Y_predict1.shape[0] + 1)]
# x_axis, y_axis = np.meshgrid(x_axis, y_axis)
# plt.pcolormesh(x_axis, y_axis, Y_predict1, cmap='flag')
# plt.xlim(1, Y_predict1.shape[1] + 1)
# plt.ylim(1, Y_predict1.shape[0] + 1)
# plt.xlabel('stability vector')
# plt.ylabel('samples')
#
# plt.subplot(133)
# plt.title('Ground truth')
# x_axis = [i + 1 for i in range(Y_validate.shape[1] + 1)]
# y_axis = [i + 1 for i in range(Y_validate.shape[0] + 1)]
# x_axis, y_axis = np.meshgrid(x_axis, y_axis)
# plt.pcolormesh(x_axis, y_axis, Y_validate, cmap='flag')
# plt.xlim(1, Y_validate.shape[1] + 1)
# plt.ylim(1, Y_validate.shape[0] + 1)
# plt.xlabel('stability vector')
# plt.tight_layout()
# plt.tick_params(axis='both', labelleft='off')
# plt.savefig(save_path + 'neural_network_unorganized', dpi=600)
#
# plt.close('all')

##############################
# reorganized training data
X_training_reorganize, Y_training_reorganize = reorganize_training_data.reorganize_training_data(training_data)

# split training data into two parts: training_sub and training_validate to assess the performance
X_training_sub = X_training_reorganize[:18000]
Y_training_sub = Y_training_reorganize[:18000]

X_validate = X_training_reorganize[18000:]
Y_validate = Y_training_reorganize[18000:]

# neural network training
hidden_layer_configs = [(48,), (96, 48, 1), (96, 96, 48, 24, 1), (96, 96, 48, 48, 24, 24, 1), (96, 96, 48, 48, 24, 24, 12, 12, 1)]

f_measures2 = []
for hidden_layer in hidden_layer_configs:
    clf = []
    # clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='logistic', solver='lbfgs')
    # clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation = 'logistic')
    # clf = MLPClassifier(hidden_layer_sizes=hidden_layer, solver='lbfgs')
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer)
    clf.fit(X_training_sub, Y_training_sub)
    Y_predict2 = clf.predict(X_validate)
    f_measure2 = f1_score(Y_validate, Y_predict2)
    f_measures2.append(f_measure2)
    print f_measure2

clf = []
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_configs[0])
clf.fit(X_training_sub, Y_training_sub)
Y_predict2 = clf.predict(X_validate)

Y_validate = Y_validate.reshape(572, 9)
Y_predict2 = Y_predict2.reshape(572, 9)

# visualize Y_predict and Y_validate(truth)
plt.figure(1, (18, 6))
plt.subplot(131)
plt.plot([1, 3, 5, 7, 9], f_measures2, 'o')
plt.ylim(0, 1)
plt.xlabel('# of hidden layers')
plt.ylabel('f-measure')

plt.subplot(132)
plt.title('prediction')
x_axis = [i + 1 for i in range(Y_predict2.shape[1] + 1)]
y_axis = [i + 1 for i in range(Y_predict2.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_predict2, cmap='flag')
plt.xlim(1, Y_predict2.shape[1] + 1)
plt.ylim(1, Y_predict2.shape[0] + 1)
plt.xlabel('stability vector')
plt.ylabel('samples')

plt.subplot(133)
plt.title('Ground truth')
x_axis = [i + 1 for i in range(Y_validate.shape[1] + 1)]
y_axis = [i + 1 for i in range(Y_validate.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_validate, cmap='flag')
plt.xlim(1, Y_validate.shape[1] + 1)
plt.ylim(1, Y_validate.shape[0] + 1)
plt.xlabel('stability vector')
plt.tight_layout()
plt.tick_params(axis='both', labelleft='off')
plt.savefig(save_path + 'neural_network_organized', dpi=600)

plt.close('all')