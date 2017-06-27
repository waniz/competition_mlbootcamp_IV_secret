import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

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

    'f_42',  # 0.904
]

exported_pipeline = ExtraTreesClassifier(max_features=0.4537270875668709, criterion='entropy', min_samples_leaf=1,
                                         min_samples_split=2, n_estimators=3138)

score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
super_y_preds = []
for seed in range(10):
    np.random.seed(seed)
    scores = model_selection.cross_val_score(exported_pipeline, X[best_columns], Y['target'], cv=12,
                                             scoring='neg_log_loss', n_jobs=3)
    print("Log Loss: %0.4f (+/- %0.3f) [seed %s]" % (scores.mean(), scores.std(), seed))

    exported_pipeline.fit(X[best_columns].as_matrix(), Y['target'].as_matrix())
    y_pred = exported_pipeline.predict_proba(score_dataset[best_columns])
    super_y_preds.append(y_pred)

answer_file = []
for pos in range(len(super_y_preds[0])):
    ans_0, ans_1, ans_2, ans_3, ans_4 = [], [], [], [], []
    for model in range(10):
        ans_0.append(super_y_preds[model][pos][0])
        ans_1.append(super_y_preds[model][pos][1])
        ans_2.append(super_y_preds[model][pos][2])
        ans_3.append(super_y_preds[model][pos][3])
        ans_4.append(super_y_preds[model][pos][4])
    row = [np.mean(ans_0), np.mean(ans_1), np.mean(ans_2), np.mean(ans_3), np.mean(ans_4)]
    answer_file.append(np.argmax(row))

pd.Series(answer_file).to_csv('data/answer.csv', index=False)
