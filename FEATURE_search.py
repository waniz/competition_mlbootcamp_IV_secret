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

    # adds
    'f_42',  # 0.904
    'f_218'  # 0.8997


]

exported_pipeline = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=2, n_estimators=100)

min_log = 1
best_feature = ''
for feature in names:
    my_columns = best_columns[:]
    my_columns.append(feature)
    if feature in best_columns:
        continue
    scores = model_selection.cross_val_score(exported_pipeline, X[my_columns], Y['target'],
                                             cv=12, scoring='neg_log_loss', n_jobs=3)
    print("Log Loss: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), feature))
    if abs(scores.mean()) < min_log:
        best_feature = feature
        min_log = abs(scores.mean())

print('Best feature:', best_feature, min_log)
