import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

"""
Check the ensemble 0.6756 results: Correct answers count: 421   = 0.6749117

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


clf1 = ExtraTreesClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=1000, n_jobs=3)
clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000, n_jobs=3)
clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000, n_jobs=3)
clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000, n_jobs=3)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4], weights=[1, 1, 1, 2], voting='soft')

# labels = ['Trees_3', 'Trees_4', 'Trees_5', 'Trees_6', 'Trees_7', 'Ensemble']
# for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], labels):
#     scores = model_selection.cross_val_score(clf, X[best_columns], Y['target'], cv=5, scoring='neg_log_loss', n_jobs=3)
#     print("Log Loss: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))

# create train and validate
X['target'] = Y['target']
x_valid_0 = pd.DataFrame(X[X['target'] == 0][90:])
x_valid_1 = pd.DataFrame((X[X['target'] == 1][900:]))
x_valid_2 = pd.DataFrame(X[X['target'] == 2][1300:])
x_valid_3 = pd.DataFrame(X[X['target'] == 3][420:])
x_valid_4 = pd.DataFrame(X[X['target'] == 4][90:])
x_valid = pd.DataFrame(pd.concat([x_valid_0, x_valid_1, x_valid_2, x_valid_3, x_valid_4], axis=0))
y_valid = pd.DataFrame()
y_valid['target'] = x_valid['target']
x_valid.drop('target', axis=1, inplace=True)
x_train_0 = pd.DataFrame(X[X['target'] == 0][:90])
x_train_1 = pd.DataFrame((X[X['target'] == 1][:900]))
x_train_2 = pd.DataFrame(X[X['target'] == 2][:1300])
x_train_3 = pd.DataFrame(X[X['target'] == 3][:420])
x_train_4 = pd.DataFrame(X[X['target'] == 4][:90])
x_train = pd.DataFrame(pd.concat([x_train_0, x_train_1, x_train_2, x_train_3, x_train_4], axis=0))
y_train = pd.DataFrame()
y_train['target'] = x_train['target']
x_train.drop('target', axis=1, inplace=True)

eclf.fit(x_train[best_columns], y_train['target'])
preds = eclf.predict(x_valid[best_columns])
print('Confusion matrix:\n')
print(confusion_matrix(y_valid['target'].values, preds))
matrix_ = confusion_matrix(y_valid['target'].values, preds)
correct_answers = matrix_[0][0] + matrix_[1][1] + matrix_[2][2] + matrix_[3][3] + matrix_[4][4]
print('Correct answers count: ', correct_answers)

# --- answer module ---
eclf.fit(X[best_columns], Y['target'])
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
y_pred = eclf.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
