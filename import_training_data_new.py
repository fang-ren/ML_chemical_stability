# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import pandas as pd
import numpy as np

def import_training_data(training_file = 'challenge_data\\challenge_data\\training_data.csv'):
    # import data into dataframe
    data = pd.read_csv(training_file)

    # get column names from the dataframe
    names = list(data)
    # delete the first two columns, which can be represented exclusively by "formulaA_elements_Number" and "formulaB_elements_Number"
    del data[names[0]]
    del data[names[1]]

    names = list(data) # update column names
    names = np.array(names)

    # extract variable matrix
    X = data.ix[:, :-1]

    # reorganize the stability vectors, the response
    vectors = []
    stabilityVec = data[names[-1]]
    for vec in stabilityVec:
        vec = vec[1:-1].split(',')
        vectors.append(vec)

    vectors = np.array(vectors)
    stabilityVec = vectors.astype(float)
    #print stabilityVec

    Y = pd.DataFrame(data = {'composition_A=0.1':stabilityVec[:,1], 'composition_A=0.2':stabilityVec[:,2],
                             'composition_A=0.3': stabilityVec[:,3],'composition_A=0.4':stabilityVec[:,4], 'composition_A=0.5':stabilityVec[:,5],
                             'composition_A=0.6': stabilityVec[:,6],'composition_A=0.7':stabilityVec[:,7], 'composition_A=0.8':stabilityVec[:,8],
                             'composition_A=0.9': stabilityVec[:,9]})
    return X, Y

#X, Y = import_training_data()