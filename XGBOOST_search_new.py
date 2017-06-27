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


def loss_func(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def logloss_func(y_true, y_pred):
    return log_loss(y_true, y_pred)


def hyperopt_train_test(hpparams):
    all_results_acc, all_results_log = [], []

    best_columns = [
        'f_138', 'f_11', 'f_96', 'f_200', 'f_32', 'f_76', 'f_79', 'f_41', 'f_83', 'f_156', 'f_131', 'f_147', 'f_187', 'f_84', 'f_182'
        ]

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X[best_columns], Y['target'])

    fold_index = 0
    starts = time.time()
    for train_index, test_index in kf.split(X[X.columns], Y['target']):
        x_train, x_test = X.as_matrix(best_columns)[train_index], X.as_matrix(best_columns)[test_index]
        y_train, y_test = Y.as_matrix()[train_index], Y.as_matrix()[test_index]

        params_est = {
            'n_estimators': round(hpparams['n_estimators']),
            'learning_rate': hpparams['eta'],
            'max_depth': hpparams['max_depth'],
            'gamma': hpparams['gamma'],
            'reg_lambda': hpparams['reg_lambda'],
            'min_child_weight': hpparams['min_child_weight'],
            'nthread': 3,
            'subsample': hpparams['subsample'],
            'colsample_bytree': hpparams['colsample_bytree'],
            'eval_metric': ['merror', 'mlogloss'],
          }

        model = xgb.XGBClassifier(**params_est)
        model.fit(x_train, y_train, verbose=True, early_stopping_rounds=30)

        y_test_pred = model.predict(x_test)
        current_acc = loss_func(y_test, y_test_pred)
        y_test_pred = model.predict_proba(x_test)
        current_log = logloss_func(y_test, y_test_pred)

        fold_index += 1
        print('      Fold %s trained: acc: %s, mlogloss: %s time: %s min' %
              (fold_index, round(current_acc, 4), round(current_log, 4), round((time.time() - starts) / 60, 2)))

        all_results_acc.append(current_acc)
        all_results_log.append(current_log)
    return np.mean(all_results_log), np.std(all_results_log), min(all_results_log), max(all_results_log), np.mean(all_results_acc)


space4dt = {
   'n_estimators': hp.uniform('n_estimators', 300, 1600),
   'max_depth': hp.choice('max_depth', (4, 5, 6, 7, 8, 9, 10, 11, 12)),
   'eta': hp.uniform('eta', 0.002, 0.2),
   'gamma': hp.uniform('gamma', 0, 1),
   'reg_lambda': hp.uniform('reg_lambda', 0.55, 1),
   'min_child_weight': hp.uniform('min_child_weight', 1.0, 2.0),
   'subsample': hp.uniform('subsample', 0.4, 1),
   'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}


def f(params):
    global log_, acc_, std_, params_, min_, max_, counter
    log, std, min_value, max_value, acc = hyperopt_train_test(params)
    counter += 1
    print_params = {}
    for key_ in params.keys():
        print_params[key_] = round(params[key_], 7)
    print(counter, round(log, 4), round(acc, 4), ' ', round(std, 4), ' ', round(max_value, 4))

    acc_.append(acc)
    log_.append(log)
    std_.append(std)
    min_.append(min_value)
    max_.append(max_value)
    params_.append(params)

    best_params = pd.DataFrame()
    best_params['log_loss'] = log_
    best_params['accuracy'] = acc_
    best_params['std'] = std_
    best_params['max_logloss'] = max_
    best_params['min_min_logloss'] = min_
    best_params['params'] = params_

    best_params.sort_values(by=['log_loss', 'accuracy', 'std'], inplace=True, ascending=False)
    best_params.to_csv('results/search_df_01_05.csv', index=False)

    return {'loss': log, 'status': STATUS_OK}


trials = Trials()
log_, acc_, std_, params_, min_, max_ = [], [], [], [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=1000, trials=None, verbose=1)
print('best:')
print(best)

