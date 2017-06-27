import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')


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
        'loss': 'log',
        'probability': True,
    }

    pipeline = SGDClassifier(**params_est)
    scores = model_selection.cross_val_score(pipeline, X[best_columns], Y['target'],
                                             cv=5, scoring='neg_log_loss', n_jobs=2)

    return scores.mean(), scores.std()


space4dt = {
   'kernel': hp.choice('kernel', ('rbf', 'rbf')),
   'probability': True,

}


def f(params):
    global log_, counter, params_, std_
    mlog, mstd = hyperopt_train_test(params)
    counter += 1

    log_.append(mlog)
    params_.append(params)
    std_.append(mstd)

    print("Log Loss: %0.4f (+/- %0.3f), %s" % (mlog, mstd, params))

    best_params = pd.DataFrame()
    best_params['mlog_loss'] = log_
    best_params['mstd'] = std_
    best_params['params'] = params_

    best_params.sort_values(by=['mlog_loss'], inplace=True, ascending=False)
    best_params.to_csv('results/search_df_16_05_best_features_KNN.csv', index=False)

    return {'loss': abs(mlog), 'status': STATUS_OK}


trials = Trials()
log_, params_, std_ = [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=5000, trials=None, verbose=1)
print('best:')
print(best)

