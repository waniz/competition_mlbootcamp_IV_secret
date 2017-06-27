import numpy as np
import pandas as pd
from copy import copy
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('../original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('../original_data/y_train.csv', names=['target'], delimiter=';')

best_columns = [
    'f_187', 'f_138', 'f_11', 'f_96', 'f_76', 'f_200', 'f_146', 'f_156', 'f_17', 'f_83', 'f_41', 'f_79', 'f_63', 'f_90',
    'f_182'
]

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    ExtraTreesClassifier(bootstrap=False, max_features=0.7500000000000001, min_samples_leaf=2, min_samples_split=7, n_estimators=100)
)
exported_pipeline.fit(X[best_columns], Y['target'])

# --- answer module ---
score_dataset = pd.read_csv('../original_data/x_test.csv', delimiter=';', names=names)
y_pred = exported_pipeline.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('../data/answer.csv', index=False)
