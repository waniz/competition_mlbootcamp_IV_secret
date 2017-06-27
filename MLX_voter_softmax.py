import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

import matplotlib.pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier, SoftmaxRegression, StackingClassifier

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

clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=1000)
clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000)
clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000)
clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000)
clf5 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000)

# 0.6713781
clf6 = ExtraTreesClassifier(max_features=0.367266672504996, criterion='entropy', min_samples_leaf=1,
                            min_samples_split=2, n_estimators=4464)
# 0.6742049
clf7 = ExtraTreesClassifier(max_features=0.4249409731271929, criterion='entropy', min_samples_leaf=1,
                            min_samples_split=2, n_estimators=3094)

clf8 = RandomForestClassifier(max_features=0.34808889858456293, criterion='entropy', min_samples_split=2,
                              n_estimators=4401)

clf9 = KNeighborsClassifier(leaf_size=11, n_neighbors=10)
clf10 = svm.SVC(kernel='rbf', probability=True)

clf11 = ExtraTreesClassifier(max_features=0.4537270875668709, criterion='entropy', min_samples_leaf=1,
                             min_samples_split=2, n_estimators=3138)

# eclf = EnsembleVoteClassifier(clfs=[clf11], weights=[1], voting='soft')
lr = SoftmaxRegression()
stacks = StackingClassifier(classifiers=[clf1, clf2, clf3, clf7, clf8, clf9, clf10, clf11], meta_classifier=lr)

labels = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9',
          'model_10', 'model_11', 'ensemble']
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, stacks], labels):
    scores = model_selection.cross_val_score(clf, X[best_columns], Y['target'], cv=12, scoring='neg_log_loss',
                                             n_jobs=3)
    print("Log Loss: %0.4f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
    if label == 'ensemble':
        X['target'] = Y['target']
        # create train and validate
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

        # --- answer module ---
        # eclf.fit(X[best_columns], Y['target'])
        score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
        y_pred = eclf.predict(score_dataset[best_columns])
        pd.Series(y_pred).to_csv('data/answer.csv', index=False)



