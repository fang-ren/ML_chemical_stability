
from import_training_data import import_training_data
from check_collinearity import check_collinearity
from reorganize_data import reorganize_data
from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

save_path = 'report//single_variable_visualization//'

X, Y = import_training_data()
#X, Y = data_cleaning(X, Y, 5)
X = check_collinearity(X)
X, Y = reorganize_data(X, Y)

sample_num = X.shape[0]
#round(sample_num*0.7,0)
X_training = X.ix[:16200, :].as_matrix()
Y_training = Y.ix[:16200, :].as_matrix()

X_test = X.ix[16200:, :].as_matrix()
Y_test = Y.ix[16200:, :].as_matrix()

rf = MLPClassifier(hidden_layer_sizes=(96, 96, 48, 24, 1))
rf.fit(X_training, Y_training)
Y_predict = rf.predict(X_test)

Y_predict = Y_predict.reshape(772, 9)
Y_test = Y_test.reshape(772, 9)

plt.subplot(121)
plt.title('prediction')
x_axis = [i + 1 for i in range(Y_predict.shape[1] + 1)]
y_axis = [i + 1 for i in range(Y_predict.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_predict, cmap='flag')
plt.xlim(1, Y_predict.shape[1] + 1)
plt.ylim(1, Y_predict.shape[0] + 1)
plt.xlabel('stability vector')
plt.ylabel('samples')

plt.subplot(122)
plt.title('test')
x_axis = [i + 1 for i in range(Y_test.shape[1] + 1)]
y_axis = [i + 1 for i in range(Y_test.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, Y_test, cmap='flag')
plt.xlim(1, Y_test.shape[1] + 1)
plt.ylim(1, Y_test.shape[0] + 1)
plt.xlabel('stability vector')
# plt.ylabel('samples')
