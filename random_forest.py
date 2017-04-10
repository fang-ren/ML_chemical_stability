# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import imp

import_training_data = imp.load_source("import_training_data", "import_training_data.py")
import_test_data = imp.load_source("import_test_data", "import_test_data.py")
reorganize_training_data = imp.load_source("reorganize_training_data", "training_data_reorganized.py")


save_path = 'report//'


training_data = import_training_data.import_training_data()

# to prune training_data
training_data = np.concatenate((training_data[:, :96], training_data[:,97:-1]), axis = 1)
X_training = training_data[:, :96]
Y_training = training_data[:, 96:]


################################
# originial training data format

# split training data into two parts: training_sub and training_validate to assess the performance
X_training_sub = X_training[:2000]
Y_training_sub = Y_training[:2000]

X_validate = X_training[2000:]
Y_validate = Y_training[2000:]

trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

f_measures1 = []
for tree in trees:
    Y_predict1 = []
    for i in range(Y_training.shape[1]):
        clf = RandomForestClassifier(n_estimators=tree)
        clf.fit(X_training_sub, Y_training_sub[:, i])
        Y_predict = clf.predict(X_validate)
        # print f_measure
        Y_predict1 = np.concatenate((Y_predict1, Y_predict))
    f_measure1 = f1_score(Y_validate.reshape(5148,), Y_predict1)
    f_measures1.append(f_measure1)
    print f_measure1

# choose 20 trees for prediction
Y_predict1 = []
for i in range(Y_training.shape[1]):
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X_training_sub, Y_training_sub[:, i])
    Y_predict = clf.predict(X_validate)
    Y_predict1 = np.concatenate((Y_predict1, Y_predict))
Y_predict1 = Y_predict1.reshape(9, 572).T

# visualize Y_predict and Y_validate(truth)
plt.figure(1, (18, 6))
plt.subplot(131)
plt.plot(trees, f_measures1, 'o')
plt.ylim(0, 1)
plt.xlabel('# of trees')
plt.ylabel('f-measure')

plt.subplot(132)
plt.title('prediction')
x_axis = [i+1 for i in range(Y_predict1.shape[1]+1)]
y_axis = [i+1 for i in range(Y_predict1.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_predict1, cmap = 'flag')
plt.xlim(1, Y_predict1.shape[1]+1)
plt.ylim(1, Y_predict1.shape[0]+1)
plt.xlabel('stability vector')
plt.ylabel('samples')

plt.subplot(133)
plt.title('Ground truth')
x_axis = [i+1 for i in range(Y_validate.shape[1]+1)]
y_axis = [i+1 for i in range(Y_validate.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_validate, cmap = 'flag')
plt.xlim(1, Y_validate.shape[1]+1)
plt.ylim(1, Y_validate.shape[0]+1)
plt.xlabel('stability vector')
plt.tight_layout()
plt.tick_params(axis='both', labelleft='off')
plt.savefig(save_path+'random_forest_unorganized', dpi = 600)


##############################
# reorganized training data
X_training_reorganize, Y_training_reorganize = reorganize_training_data.reorganize_training_data(training_data)

# split training data into two parts: training_sub and training_validate to assess the performance
X_training_sub = X_training_reorganize[:18000]
Y_training_sub = Y_training_reorganize[:18000]

X_validate = X_training_reorganize[18000:]
Y_validate = Y_training_reorganize[18000:]

trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
f_measures2 = []
for tree in trees:
    clf = RandomForestClassifier(n_estimators=tree)
    clf.fit(X_training_sub, Y_training_sub)
    Y_predict2 = clf.predict(X_validate)
    f_measure2 = f1_score(Y_validate, Y_predict2)
    f_measures2.append(f_measure2)
    print f_measure2

# choose 40 trees for prediction
clf = RandomForestClassifier(n_estimators=40)
clf.fit(X_training_sub, Y_training_sub)
Y_predict2 = clf.predict(X_validate)

Y_validate = Y_validate.reshape(572, 9)
Y_predict2 = Y_predict2.reshape(572, 9)

# visualize Y_predict and Y_validate(truth)
plt.figure(2, (18, 6))
plt.subplot(131)
plt.plot(trees, f_measures2, 'o')
plt.ylim(0, 1)
plt.xlabel('# of trees')
plt.ylabel('f-measure')

plt.subplot(132)
plt.title('prediction')
x_axis = [i+1 for i in range(Y_predict2.shape[1]+1)]
y_axis = [i+1 for i in range(Y_predict2.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_predict2, cmap = 'flag')
plt.xlim(1, Y_predict2.shape[1]+1)
plt.ylim(1, Y_predict2.shape[0]+1)
plt.xlabel('stability vector')
plt.ylabel('samples')

plt.subplot(133)
plt.title('Ground truth')
x_axis = [i+1 for i in range(Y_validate.shape[1]+1)]
y_axis = [i+1 for i in range(Y_validate.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_validate, cmap = 'flag')
plt.xlim(1, Y_validate.shape[1]+1)
plt.ylim(1, Y_validate.shape[0]+1)
plt.xlabel('stability vector')
plt.tight_layout()
plt.tick_params(axis='both', labelleft='off')
plt.savefig(save_path+'random_forest_organized', dpi = 600)

plt.close('all')