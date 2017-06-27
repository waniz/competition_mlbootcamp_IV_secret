import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import warnings


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

rfe_columns = [
    'f_0', 'f_4', 'f_11', 'f_21', 'f_23', 'f_24', 'f_35', 'f_36', 'f_54', 'f_61', 'f_63',
    'f_66', 'f_71', 'f_73', 'f_74', 'f_87', 'f_91', 'f_95', 'f_96', 'f_98', 'f_105',
    'f_120', 'f_134', 'f_138', 'f_156', 'f_159', 'f_165', 'f_173', 'f_182', 'f_193',
]

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X.as_matrix(X.columns), Y.as_matrix(['target']), test_size=0.05)

# # second best
# params = {
#     'n_estimators': 705,
#     'learning_rate': 0.017206202378333424,
#     'max_depth': 11,
#     'gamma': 0.13753937460231447,
#     'reg_lambda': 0.8361666020583968,
#     'min_child_weight': 1.4499742433479736,
#     'nthread': 4,
#     'subsample': 0.7883388932106451,
# }

# first best
# params = {
#     'n_estimators': 716,
#     'learning_rate': 0.06506482675644681,
#     'max_depth': 12,
#     'gamma': 0.21150398926264405,
#     'reg_lambda': 0.8763461291852132,
#     'min_child_weight': 1.4941069914141007,
#     'nthread': 4,
#     'subsample': 0.8993501954575421,
# }

# check


params = {
    'n_estimators': 497,
    'learning_rate': 0.013361347883235689,
    'max_depth': 7,
    'gamma': 0.9306761276819936,
    'reg_lambda': 0.7180114043991777,
    'min_child_weight': 1.5815243908636745,
    'nthread': 2,
    'subsample': 0.9227732809767635,
    'silent': 0,
    'colsample_bytree': 0.8007837387285838,
}


model_test = xgb.XGBClassifier(**params)
model_test.fit(x_train, y_train, verbose=True)
y_pred = model_test.predict(x_test)

metrics = accuracy_score(y_test, y_pred)
print('\nAccuracy on test-part of train score: %s' % round(metrics, 4))
print('Classification report:')
print(classification_report(y_test, y_pred))

# --- answer module ---
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
y_pred = model_test.predict(score_dataset.as_matrix(score_dataset.columns))
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
