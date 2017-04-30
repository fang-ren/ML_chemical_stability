# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import pandas as pd
import numpy as np

def import_test_data(test_file = '..\\data\\data\\test_data.csv'):
    # import data into dataframe
    data = pd.read_csv(test_file)

    # get column names from the dataframe
    names = list(data)

    # delete the first two columns, which can be represented exclusively by "formulaA_elements_Number" and "formulaB_elements_Number"
    del data[names[0]]
    del data[names[1]]
    return data

# X_test = import_test_data()
# print X_test.shape