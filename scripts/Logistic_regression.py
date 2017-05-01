
from import_training_data import import_training_data
from check_collinearity import check_collinearity
from reorganize_data import reorganize_data
from sklearn.model_selection import train_test_split
from data_cleaning import data_cleaning
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

save_path = '..//report//'


X, Y = import_training_data()
#X, Y = data_cleaning(X, Y, 5)
X = check_collinearity(X)
X, y = reorganize_data(X, Y)
y = y.ix[:,0]
names = list(X)
print X.shape, y.shape

# split the training data into training and test set
X_training, X_test, y_training, y_val = train_test_split(X, y, train_size= 0.75,random_state=0)

# correct the skewness
oversampler=SMOTE(random_state=0)
X_training, y_training = oversampler.fit_sample(X_training,y_training)

lg = LogisticRegression()
lg.fit(X_training, y_training)
y_predict = lg.predict(X_test)
score = accuracy_score(y_val, y_predict)
print score


y_predict = y_predict.reshape(len(y_val)/9, 9)
y_val = y_val.reshape(len(y_val)/9, 9)

plt.subplot(121)
plt.title('prediction')
x_axis = [i + 1 for i in range(y_predict.shape[1] + 1)]
y_axis = [i + 1 for i in range(y_predict.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, y_predict, cmap='viridis')
plt.xlim(1, y_predict.shape[1] + 1)
plt.ylim(1, y_predict.shape[0] + 1)
plt.xlabel('stability vector')
plt.ylabel('samples')

plt.subplot(122)
plt.title('test')
x_axis = [i + 1 for i in range(y_val.shape[1] + 1)]
y_axis = [i + 1 for i in range(y_val.shape[0] + 1)]
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
plt.pcolormesh(x_axis, y_axis, y_val, cmap='viridis')
plt.xlim(1, y_val.shape[1] + 1)
plt.ylim(1, y_val.shape[0] + 1)
plt.xlabel('stability vector')
# plt.ylabel('samples')
plt.savefig(save_path + 'logistic_regression_results.png')
plt.close('all')