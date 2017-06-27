import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import warnings


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

X['target'] = Y['target']

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)

# create train and validate
x_valid_0 = pd.DataFrame(X[X['target'] == 0][100:])
x_valid_1 = pd.DataFrame((X[X['target'] == 1][1000:]))
x_valid_2 = pd.DataFrame(X[X['target'] == 2][1400:])
x_valid_3 = pd.DataFrame(X[X['target'] == 3][520:])
x_valid_4 = pd.DataFrame(X[X['target'] == 4][100:])
x_valid = pd.DataFrame(pd.concat([x_valid_0, x_valid_1, x_valid_2, x_valid_3, x_valid_4], axis=0))

y_valid = pd.DataFrame()
y_valid['target'] = x_valid['target']
x_valid.drop('target', axis=1, inplace=True)
print(x_valid.shape)

x_train_0 = pd.DataFrame(X[X['target'] == 0][:100])
x_train_1 = pd.DataFrame((X[X['target'] == 1][:1000]))
x_train_2 = pd.DataFrame(X[X['target'] == 2][:1400])
x_train_3 = pd.DataFrame(X[X['target'] == 3][:520])
x_train_4 = pd.DataFrame(X[X['target'] == 4][:100])
x_train = pd.DataFrame(pd.concat([x_train_0, x_train_1, x_train_2, x_train_3, x_train_4], axis=0))

y_train = pd.DataFrame()
y_train['target'] = x_train['target']
x_train.drop('target', axis=1, inplace=True)
print(x_train.shape)

search_features = [
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

# d_train = xgb.DMatrix(x_train[search_features], label=y_train)
d_train = xgb.DMatrix(X[search_features], label=Y)

params = {
    'objective': 'multi:softmax',
    'eta': 0.016,
    'silent': 1,
    'nthread': 2,
    'eval_metric': ['merror', 'mlogloss'],
    'num_class': 5,

    # 'gamma': 1,
}

params_est = {
    'n_estimators': 506,
    'learning_rate': 0.0025437050853091394,
    'max_depth': 11,
    'gamma': 0.7559648314628407,
    'reg_lambda': 0.9623897097994525,
    'min_child_weight': 1.2456145312471927,
    'nthread': 3,
    'subsample': 0.7277237062282782,
    'colsample_bytree': 0.7303803887107063,
    'eval_metric': ['merror', 'mlogloss'],
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 5,
}

# 0.934183	0.1626347439	0.3746006
params_est_1 = {
    'n_estimators': 1430,
    'learning_rate': 0.002191956964160524,
    'max_depth': 12,
    'gamma': 0.23461710260001806,
    'reg_lambda': 0.7454051561839893,
    'min_child_weight': 1.1122743531694044,
    'nthread': 3,
    'subsample': 0.9128597857496583,
    'colsample_bytree': 0.5237051940747892,
    'eval_metric': ['merror', 'mlogloss'],
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 5,
}

# 0.9329184	0.1641284481	0.3766082
params_est_2 = {
    'n_estimators': 1765,
    'learning_rate': 0.0020246720412184505,
    'max_depth': 12,
    'gamma': 0.10327108747854377,
    'reg_lambda': 0.7803780476359986,
    'min_child_weight': 1.1321108946029863,
    'nthread': 3,
    'subsample': 0.9502439118345728,
    'colsample_bytree': 0.4522739447988407,
    'eval_metric': ['merror', 'mlogloss'],
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 5,
}

# 0.9332078	0.1656280278	0.371165
params_est_3 = {
    'n_estimators': 2025,
    'learning_rate': 0.00462301254497432,
    'max_depth': 14,
    'gamma': 0.11279367285384101,
    'reg_lambda': 0.7466601456278856,
    'min_child_weight': 1.103929531774299,
    'nthread': 3,
    'subsample': 0.9431190671196655,
    'colsample_bytree': 0.4706028872905275,
    'eval_metric': ['merror', 'mlogloss'],
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 5,
}

watchlist = [(d_train, 'train')]
model = xgb.train(params, d_train, num_boost_round=3400, evals=watchlist, early_stopping_rounds=100)
plot_importance(model)
plt.show()

# --- answer module ---
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
y_pred = model.predict(xgb.DMatrix(score_dataset[search_features]))
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
