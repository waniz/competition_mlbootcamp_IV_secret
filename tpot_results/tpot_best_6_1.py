import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('../original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('../original_data/y_train.csv', names=['target'], delimiter=';')

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

exported_pipeline = ExtraTreesClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=100)

exported_pipeline.fit(X[best_columns], Y['target'])

# --- answer module ---
score_dataset = pd.read_csv('../original_data/x_test.csv', delimiter=';', names=names)
y_pred = exported_pipeline.predict(score_dataset[best_columns])
pd.Series(y_pred).to_csv('../data/answer.csv', index=False)
