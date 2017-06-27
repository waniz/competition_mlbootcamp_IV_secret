import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
from xgboost import plot_importance, plot_tree
import warnings


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

# names = ['f_' + str(i) for i in range(223)]
# X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)

X = pd.read_csv('data/x_train_regressor_0.csv')
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)

rfe_columns = [
    'f_0', 'f_4', 'f_11', 'f_21', 'f_23', 'f_24', 'f_35', 'f_36', 'f_54', 'f_61', 'f_63',
    'f_66', 'f_71', 'f_73', 'f_74', 'f_87', 'f_91', 'f_95', 'f_96', 'f_98', 'f_105',
    'f_120', 'f_134', 'f_138', 'f_156', 'f_159', 'f_165', 'f_173', 'f_182', 'f_193',
    'xgb_regressor',
]

x_train, x_test, y_train, y_test = train_test_split(X.as_matrix(rfe_columns), Y.as_matrix(), test_size=0.053)

model_test = xgb.XGBClassifier()
model_test.fit(x_train, y_train, verbose=True)
plot_importance(model_test)
plt.show()
y_pred = model_test.predict(x_test)

metrics = accuracy_score(y_test, y_pred)
print('\nAccuracy on test-part of train score: %s' % round(metrics, 4))
print('Classification report:')
print(classification_report(y_test, y_pred))

