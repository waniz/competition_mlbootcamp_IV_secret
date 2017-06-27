import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('../original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('../original_data/y_train.csv', names=['target'], delimiter=';')

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


# -0.8613
exported_pipeline = RandomForestClassifier(max_features=0.34808889858456293, criterion='entropy',
                                           min_samples_split=2, n_estimators=4401, n_jobs=4)

exported_pipeline.fit(X[best_columns], Y['target'])

# --- answer module ---
score_dataset = pd.read_csv('../original_data/x_test.csv', delimiter=';', names=names)
y_pred = exported_pipeline.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('../data/answer.csv', index=False)
