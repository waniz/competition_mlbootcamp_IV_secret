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

x_train, x_test, y_train, y_test = train_test_split(X.as_matrix(rfe_columns), Y.as_matrix(), test_size=0.1)

params = {
    'n_estimators': 3000,
    # 'learning_rate': 0.006012166011219063,
    # 'max_depth': 7,
    # 'gamma': 0.09674730440257719,
    # 'reg_lambda': 0.8424464856859664,
    # 'min_child_weight': 1.375568973652835,
    # 'nthread': 4,
    # 'subsample': 0.7347078551244224,
    # # 'colsample_bytree': 0.5627386560255774,
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

y_pred = model_test.predict(score_dataset.as_matrix(rfe_columns))
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
