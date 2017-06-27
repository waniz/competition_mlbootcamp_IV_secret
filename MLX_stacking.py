import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

best_columns = [
    'f_187', 'f_138', 'f_11', 'f_96', 'f_76', 'f_200', 'f_146', 'f_156', 'f_17', 'f_83', 'f_41', 'f_79', 'f_63', 'f_90',
    'f_182'
]

# 0.6565
clf4 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=100)

# OK clf9
clf9 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=115)

lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[clf4, clf9],
                            use_probas=True,
                            meta_classifier=lr,
                            random_state=42)

print('3-fold cross validation:\n')

for clf, label in zip([clf4, clf9, sclf],
                      ['Tree_4',
                       'Tree_9',
                       'StackingCVClassifier']):

    scores = model_selection.cross_val_score(clf, X[best_columns].as_matrix(), Y['target'].as_matrix(),
                                             cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.3f) [%s]"
          % (scores.mean(), scores.std(), label))
