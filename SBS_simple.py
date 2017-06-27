import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

estimator = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=100)

sfs1 = SFS(estimator,
           k_features=(10, 40),
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5,
           n_jobs=4)

sfs1 = sfs1.fit(X[X.columns].as_matrix(), Y['target'].values)

results = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

# (96, 97, 98, 131, 200, 138, 11, 76, 115, 83, 212, 182, 187, 156)
# 0.642879680873

# (96, 98, 131, 138, 11, 76, 43, 209, 115, 182, 29)
# 0.638581676053


print(sfs1.subsets_)
print(sfs1.k_feature_idx_)
print(sfs1.k_score_)
