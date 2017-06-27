import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_split=2, n_estimators=100)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)

# 0.9118
best_columns = [
    'f_138',
    'f_11',
    'f_96',
    'f_200',
    'f_32',
    'f_76',
    # 'f_79',
    'f_41', 'f_83', 'f_156', 'f_131',
    # 'f_147',
    # 'f_187',
    'f_84', 'f_182', 'f_108',
    'f_17',
]
