import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import warnings
import time

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)


names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)


def hyperopt_train_test(hpparams):

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

    params_est = {
        'n_estimators': round(hpparams['n_estimators']),
        'learning_rate': hpparams['eta'],
        'max_depth': hpparams['max_depth'],
        'gamma': hpparams['gamma'],
        'reg_lambda': hpparams['reg_lambda'],
        'min_child_weight': hpparams['min_child_weight'],
        'nthread': 1,
        'subsample': hpparams['subsample'],
        'colsample_bytree': hpparams['colsample_bytree'],
        'eval_metric': ['merror', 'mlogloss'],
        'silent': 1,
        'objective': 'multi:softmax',
        'num_class': 5,
      }

    d_train = xgb.DMatrix(X[best_columns], label=Y)
    model = xgb.cv(params_est, d_train, 3400, nfold=5, stratified=True, early_stopping_rounds=300)

    print(min(model['test-mlogloss-mean']), min(model['test-merror-mean']), np.std(model['test-mlogloss-mean']))

    return min(model['test-mlogloss-mean']), min(model['test-merror-mean']), np.std(model['test-mlogloss-mean'])


space4dt = {
   'n_estimators': hp.uniform('n_estimators', 1000, 3500),
   'max_depth': hp.choice('max_depth', (11, 12, 13, 14, 15, 16)),
   'eta': hp.uniform('eta', 0.0001, 0.03),
   'gamma': hp.uniform('gamma', 0, 0.5),
   'reg_lambda': hp.uniform('reg_lambda', 0.6, 0.8),
   'min_child_weight': hp.uniform('min_child_weight', 1.0, 1.2),
   'subsample': hp.uniform('subsample', 0.9, 1),
   'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.6),
}


def f(params):
    global log_, acc_, counter, params_, std_
    mlog, merr, mstd = hyperopt_train_test(params)
    counter += 1

    acc_.append(merr)
    log_.append(mlog)
    params_.append(params)
    std_.append(mstd)

    best_params = pd.DataFrame()
    best_params['mlog_loss'] = log_
    best_params['mstd'] = std_
    best_params['accuracy'] = acc_
    best_params['params'] = params_

    best_params.sort_values(by=['mlog_loss'], inplace=True, ascending=False)
    best_params.to_csv('results/search_df_14_05_best_features_XGBOOST.csv', index=False)

    return {'loss': mlog, 'status': STATUS_OK}


trials = Trials()
log_, acc_, params_, std_ = [], [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=5000, trials=None, verbose=1)
print('best:')
print(best)

