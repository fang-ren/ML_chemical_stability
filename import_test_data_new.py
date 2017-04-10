# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import pandas as pd
import numpy as np

def import_test_data(test_file = 'challenge_data\\challenge_data\\test_data.csv'):
    # import data into dataframe
    data = pd.read_csv(test_file)

    # get column names from the dataframe
    names = list(data)

    # delete the first two columns, which can be represented exclusively by "formulaA_elements_Number" and "formulaB_elements_Number"
    del data[names[0]]
    del data[names[1]]
    names = list(data) # update column names
    names = np.array(names)
    return names, data