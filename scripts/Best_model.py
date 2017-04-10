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
reorganize_test_data = imp.load_source("reorganize_test_data", "test_data_reorganized.py")

save_path = 'report//'


training_data = import_training_data.import_training_data()

# to prune training_data
training_data = np.concatenate((training_data[:, :96], training_data[:,97:-1]), axis = 1)
X_training = training_data[:, :96]
Y_training = training_data[:, 96:]

# import test data
test_data = import_test_data.import_test_data()

# reorganized training data
X_training_reorganize, Y_training_reorganize = reorganize_training_data.reorganize_training_data(training_data)

# reorganized test data
test_data_reorganize = reorganize_test_data.reorganize_test_data(test_data)

# choose 40 trees for prediction
clf = RandomForestClassifier(n_estimators=40)
clf.fit(X_training_reorganize, Y_training_reorganize)
Y_predict = clf.predict(test_data_reorganize)

Y_predict = Y_predict.reshape(749, 9)

# re-format predicted result into stability vectors
vectors = [['stability vector'],]
for row in range(Y_predict.shape[0]):
    # print row, Y_predict[row]
    vector = [1,]
    for v in Y_predict[row]:
        # print v
        vector.append(v)
    vector.append(1)
    vectors.append(vector)

import csv
with open('C:\Research_FangRen\\Python codes\\Citrine_challenge\\challenge_data\\challenge_data\\test_data.csv', 'rb') as csvfile:
    test_result = csv.reader(csvfile, delimiter=',')
    rows = []
    i = 0
    for row in test_result:
        row = row + [vectors[i]]
        i += 1
        rows.append(row)


with open('C:\\Research_FangRen\\Python codes\\Citrine_challenge\\report\\results.csv', 'a') as csvoutput:
    writer = csv.writer(csvoutput, delimiter=',', lineterminator='\n')
    for row in rows:
        writer.writerow(row)
    csvoutput.close()
