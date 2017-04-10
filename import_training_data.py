# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np

def import_training_data(training_file = 'challenge_data\\challenge_data\\training_data.csv'):
    f = open(training_file, 'r')
    lines = f.read().split("\n")
    training_data = []

    for line in lines:
        if lines.index(line) == 0: # add other needed checks to skip titles
            header = line.split(",")
            # print np.array(header).T
        else:
            line = line.split(",")
            element1 = line[0]
            element2 = line[1]
            descriptors = line[2:-11]
            descriptors = [float(i) for i in descriptors]

            stability_vector = [line[-11][2:]] + line[-10:-1] + [line[-1][:-2]]
            stability_vector = [float(i) for i in stability_vector]

            line = descriptors + stability_vector
            training_data.append(line)

    training_data = np.array(training_data)
    return training_data


## to run
# training_data = import_training_data()
