from import_training_data_new import import_training_data
from check_collinearity import check_collinearity
from data_cleaning import data_cleaning


save_path = 'report//'

X, Y = import_training_data()
#X, Y = data_cleaning(X, Y, 5)
X = check_collinearity(X)
summary = X.describe()

summary.to_csv(save_path + 'variable_summary.csv')
