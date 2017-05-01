
from import_training_data import import_training_data
from check_collinearity import check_collinearity
from reorganize_data import reorganize_data
from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os.path



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

# random forest parameter
params_rf = {
    'n_jobs': 1,
    'n_estimators': 1600,
    'warm_start': True,
    'max_features': 0.3,
    'max_depth': 9,
    'min_samples_leaf': 2,
    'random_state' : 0,
    'verbose': 0
}

# random forest
rf = RandomForestClassifier(**params_rf)
rf.fit(X_training, y_training)
y_predict = rf.predict(X_test)
score = accuracy_score(y_val, y_predict)
print score

y_predict = y_predict.reshape(len(y_val)/9, 9)
y_val = y_val.reshape(len(y_val)/9, 9)

save_path = '..\\report\\'
if not os.path.exists(save_path):
    os.mkdir(save_path)

plt.figure(1)
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
plt.savefig(os.path.join(save_path, 'random_forest_results'))

feature_importances = rf.feature_importances_
plt.figure(2)
plt.scatter(range(len(names)), feature_importances, c = feature_importances, cmap = 'jet')
plt.xticks(range(len(names)), names, rotation = 90)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'random_forest_feature_importances'))
plt.close('all')