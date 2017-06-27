import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)
plt.style.use('ggplot')


names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

# default features:
# Log Loss: -0.920 (+/- 0.042) [Trees_3]   -0.883 (500t)  -0.880 (1500t)
# Log Loss: -0.919 (+/- 0.034) [Trees_4]   -0.898 (500t)  -0.896 (1500t)
# Log Loss: -0.923 (+/- 0.044) [Trees_5]   -0.881 (500t)  -0.879 (1500t)
# Log Loss: -0.920 (+/- 0.034) [Trees_6]   -0.888 (500t)  -0.888 (1500t)
# Log Loss: -0.903 (+/- 0.025) [Trees_7]   -0.891 (500t)  -0.889 (1500t)
# Log Loss: -0.888 (+/- 0.010) [Ensemble]  -0.885         -0.885

"""
Notes:
 - f_17 - f_84 = -0.887
 - f_17 - f_32 = -0.891 (0.901)

 f_11 - 0.14 * f_0 ~ target
"""

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

clf3 = ExtraTreesClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
clf4 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=100)
clf5 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
# current best
clf6 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

# check the amount of trees
clf7 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

# gives + 0.0016
eclf = EnsembleVoteClassifier(clfs=[clf3, clf4, clf5, clf6, clf7], weights=[1, 1, 1, 1, 1], voting='soft')

labels = ['Trees_3', 'Trees_4', 'Trees_5', 'Trees_6', 'Trees_7', 'Ensemble']
for clf, label in zip([clf3, clf4, clf5, clf6, clf7, eclf], labels):
    scores = model_selection.cross_val_score(clf, X[best_columns], Y['target'], cv=5, scoring='neg_log_loss')
    print("Log Loss: %0.4f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))

# # --- answer module ---
# eclf.fit(X[best_columns], Y['target'])
# score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
# y_pred = eclf.predict(score_dataset[best_columns])
# pd.Series(y_pred).to_csv('data/answer.csv', index=False)
