# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np

def import_test_data(test_file = 'challenge_data\\challenge_data\\test_data.csv'):
    f = open(test_file, 'r')
    lines = f.read().split("\n")
    test_data = []

    for line in lines:
        if lines.index(line) == 0: # add other needed checks to skip titles
            header = line.split(",")
        else:
            line = line.split(",")
            element1 = line[0]
            element2 = line[1]
            descriptors = line[2:-11]
            descriptors = [float(i) for i in descriptors]

            stability_vector = [line[-11][2:]] + line[-10:-1] + [line[-1][:-2]]
            stability_vector = [float(i) for i in stability_vector]

            line = descriptors + stability_vector
            test_data.append(line)

    test_data = np.array(test_data)
    return test_data