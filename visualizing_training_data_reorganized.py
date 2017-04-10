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
reorganize_training_data = imp.load_source("reorganize_training_data", "training_data_reorganized.py")

save_path = 'report//'

training_data = import_training_data.import_training_data()
X_training, Y_training = reorganize_training_data.reorganize_training_data(training_data)
Y_training = np.concatenate(([Y_training], [Y_training])).T

# visualize training_data
plt.figure(1, (7,6))

ax1 = plt.subplot2grid((1,6), (0,0), colspan=5)
plt.title('X')
x_axis = [i+1 for i in range(X_training.shape[1]+1)]
y_axis = [i+1 for i in range(X_training.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
ax1.pcolormesh(x_axis, y_axis, X_training, cmap = 'flag')
plt.xlim(1, X_training.shape[1]+1)
plt.ylim(1, X_training.shape[0]+1)
plt.xlabel('features')
plt.ylabel('samples')

ax1 = plt.subplot2grid((1,6), (0,5))
plt.title('y')
x_axis = [0, 1]
y_axis = [i+1 for i in range(Y_training.shape[0]+1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_training, cmap = 'flag')
# plt.xlim(1, Y_training.shape[1]+1)
plt.ylim(1, Y_training.shape[0]+1)
plt.xlabel('stability')

plt.tight_layout()
plt.tick_params(axis='both', labelleft='off', labelbottom = 'off', bottom = 'off', top = 'off')
plt.savefig(save_path+'training_data_reorganized', dpi = 600)

print X_training.shape, Y_training.shape