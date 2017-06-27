import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import warnings
import time

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)


def loss_func(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def hyperopt_train_test(hpparams):
    all_results = []

    rfe_columns = [
        'f_0', 'f_4', 'f_11', 'f_21', 'f_23', 'f_24', 'f_35', 'f_36', 'f_54', 'f_61', 'f_63',
        'f_66', 'f_71', 'f_73', 'f_74', 'f_87', 'f_91', 'f_95', 'f_96', 'f_98', 'f_105',
        'f_120', 'f_134', 'f_138', 'f_156', 'f_159', 'f_165', 'f_173', 'f_182', 'f_193',
    ]

    kf = StratifiedKFold(n_splits=4, shuffle=True)
    kf.get_n_splits(X[rfe_columns], Y['target'])

    fold_index = 0
    starts = time.time()
    for train_index, test_index in kf.split(X[rfe_columns], Y['target']):
        print(train_index)
        x_train, x_test = X.as_matrix(rfe_columns)[train_index], X.as_matrix(rfe_columns)[test_index]
        y_train, y_test = Y.as_matrix()[train_index], Y.as_matrix()[test_index]

        # print('Train Kfold shape:', x_train.shape, 'Test Kfold shape:', x_test.shape)

        params_est = {
            'n_estimators': round(hpparams['n_estimators']),
            'learning_rate': hpparams['eta'],
            'max_depth': hpparams['max_depth'],
            'gamma': hpparams['gamma'],
            'reg_lambda': hpparams['reg_lambda'],
            'min_child_weight': hpparams['min_child_weight'],
            'nthread': 4,
            'subsample': hpparams['subsample'],
          }

        # print('  Folds started:')

        model = xgb.XGBClassifier(**params_est)
        model.fit(x_train, y_train, verbose=True)

        y_test_pred = model.predict(x_test)
        current_res = loss_func(y_test, y_test_pred)

        fold_index += 1
        print('      Fold %s trained: res: %s, time: %s min' %
              (fold_index, round(current_res, 4), round((time.time() - starts) / 60, 2)))

        all_results.append(current_res)
    return np.mean(all_results), np.std(all_results), min(all_results), max(all_results)


space4dt = {
   'n_estimators': hp.uniform('n_estimators', 100, 1600),
   'max_depth': hp.choice('max_depth', (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
   'eta': hp.uniform('eta', 0.005, 0.5),
   'gamma': hp.uniform('gamma', 0, 1),
   'reg_lambda': hp.uniform('reg_lambda', 0.55, 1),
   'min_child_weight': hp.uniform('min_child_weight', 1.2, 3.0),
   'subsample': hp.uniform('subsample', 0.6, 1),
   # 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}


def f(params):
    global acc_, std_, params_, counter, min_, max_
    acc, std, min_value, max_value = hyperopt_train_test(params)
    counter += 1
    print_params = {}
    for key_ in params.keys():
        print_params[key_] = round(params[key_], 7)
    print(counter, round(acc, 4), ' ', round(std, 4), ' ', round(max_value, 4), print_params)

    acc_.append(acc)
    std_.append(std)
    min_.append(min_value)
    max_.append(max_value)
    params_.append(params)

    best_params = pd.DataFrame()
    best_params['accuracy'] = acc_
    best_params['std'] = std_
    best_params['max_accuracy'] = max_
    best_params['min_accuracy'] = min_
    best_params['params'] = params_

    best_params.sort_values(by=['accuracy', 'std', 'max_accuracy'], inplace=True, ascending=False)
    best_params.to_csv('results/search_df_23_04.csv', index=False)

    return {'loss': 1/acc, 'status': STATUS_OK}


trials = Trials()
acc_, std_, params_, min_, max_ = [], [], [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=1000, trials=None, verbose=1)
print('best:')
print(best)

