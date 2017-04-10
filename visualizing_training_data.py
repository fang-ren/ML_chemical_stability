# -*- coding: utf-8 -*-
"""
Created on Wed 12/14/16

@author: fangren

"""

import numpy as np
import matplotlib.pyplot as plt
import imp


import_training_data = imp.load_source("import_training_data", "import_training_data.py")
import_test_data = imp.load_source("import_test_data", "import_test_data.py")
save_path = 'report//'

# read training data
X_training = import_training_data.import_training_data()[:, :96]
Y_training = import_training_data.import_training_data()[:, 96:]

# visualize training_data
plt.figure(1, (12,6))


plt.subplot(121)
plt.title('X')
x_axis = [i+1 for i in range(X_training.shape[1]+1)]
y_axis = [i+1 for i in range(X_training.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, X_training, cmap = 'flag')
plt.xlim(1, X_training.shape[1]+1)
plt.ylim(1, X_training.shape[0]+1)
plt.xlabel('features')
plt.ylabel('samples')

plt.subplot(122)
plt.title('y')
x_axis = [i+1 for i in range(Y_training.shape[1]+1)]
y_axis = [i+1 for i in range(Y_training.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_training, cmap = 'flag')
plt.xlim(1, Y_training.shape[1]+1)
plt.ylim(1, Y_training.shape[0]+1)
plt.xlabel('stability vector')

plt.tight_layout()
plt.tick_params(axis='both', labelleft='off')
plt.savefig(save_path+'training_data', dpi = 600)

print X_training.shape, Y_training.shape
