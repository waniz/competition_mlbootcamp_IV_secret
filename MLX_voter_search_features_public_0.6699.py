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

clf1 = ExtraTreesClassifier(max_features=0.367266672504996, criterion='entropy',
                            min_samples_leaf=1, min_samples_split=2,
                            n_estimators=4464)
clf2 = ExtraTreesClassifier(max_features=0.42832108163797955, criterion='entropy',
                            min_samples_leaf=1, min_samples_split=2,
                            n_estimators=4336)
clf3 = ExtraTreesClassifier(max_features=0.5443589662774958, criterion='entropy',
                            min_samples_leaf=1, min_samples_split=2,
                            n_estimators=4094)
clf4 = ExtraTreesClassifier(max_features=0.5110350963098178, criterion='entropy',
                            min_samples_leaf=1, min_samples_split=2,
                            n_estimators=4256)
clf5 = ExtraTreesClassifier(max_features=0.4810819193211817, criterion='entropy',
                            min_samples_leaf=1, min_samples_split=2,
                            n_estimators=3251)

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1], voting='soft')

# labels = ['Trees_1', 'Trees_2', 'Trees_3', 'Trees_4', 'Trees_5', 'Ensemble']
# for clf, label in zip([clf1, clf2, clf3], labels):
#     scores = model_selection.cross_val_score(clf, X[best_columns], Y['target'], cv=5, scoring='neg_log_loss', n_jobs=4)
#     print("Log Loss: %0.4f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))

# --- answer module ---
print('Fit')
eclf.fit(X[best_columns], Y['target'])
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
print('Predict')
y_pred = eclf.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
