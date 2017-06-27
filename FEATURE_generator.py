import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')


best_columns = [
    'f_138',
    'f_11',
    'f_96',
    'f_200',
    'f_76',
    'f_41',
    'f_83',
    'f_156',
    'f_131',
    'f_84',
    'f_182',
]

print(X[best_columns])

X_ = X[best_columns]
print('\nBefore transformation: ', X_.shape)

for i1, col1 in enumerate(best_columns):
    for i2, col2 in enumerate(best_columns):
        if col1 == col2:
            continue
        X_['%s_%s_1' % (col1, col2)] = X_[col1] / (X_[col2] + 1)
        X_['%s_%s_2' % (col1, col2)] = X_[col1] * X_[col2]

print('Final transformation: ', X_.shape)
