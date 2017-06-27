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

best_columns = [  # 0.627 // 0.628
    'f_138',   # 0.619 // 0.622
    'f_11',   # 0.622 // 0.620
    'f_96',   # 0.622 // 0.626
    'f_200',  # 0.623 // 0.625
    'f_32',   # 0.628 // 0.624
    'f_76',   # 0.625 // 0.623
    'f_41',   # 0.627 // 0.630
    'f_83',   # 0.627 // 0.628
    'f_156',  # 0.625 // 0.617
    'f_131',  # 0.627 // 0.624
    'f_84',   # 0.632 // 0.629
    'f_182',  # 0.625 // 0.627
    # 'f_108',  # 0.634 // 0.633
    'f_17',   # may be, need to check 0.628
]

clf3 = ExtraTreesClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
clf4 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=100)
clf5 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
# current best
clf6 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

eclf = EnsembleVoteClassifier(clfs=[clf3, clf4, clf5, clf6], weights=[1, 1, 1, 1], voting='soft')

labels = ['Trees_3', 'Trees_4', 'Trees_5', 'Trees_6',
          'Ensemble']

for clf, label in zip([clf3, clf4, clf5, clf6, eclf], labels):

    scores = model_selection.cross_val_score(clf, X[best_columns], Y['target'],
                                             cv=4, scoring='neg_log_loss')
    print("Log Loss: %0.3f (+/- %0.3f) [%s]"
          % (scores.mean(), scores.std(), label))

# --- answer module ---
eclf.fit(X[best_columns], Y['target'])
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
y_pred = eclf.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
