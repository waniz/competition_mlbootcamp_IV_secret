import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from mlxtend.classifier import EnsembleVoteClassifier

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


class GetThatEnsemble:

    def __init__(self, cpu):
        self.names = ['f_' + str(i) for i in range(223)]
        self.X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=self.names)
        self.Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

        # feature generator block:
        self.X['mul_0'] = self.X['f_138'] * self.X['f_96']
        self.X['mul_1'] = self.X['f_138'] * self.X['f_156']
        self.X['mul_2'] = self.X['f_11'] * self.X['f_200']
        self.X['mul_3'] = self.X['f_96'] * self.X['f_83']
        self.X['mul_4'] = self.X['f_200'] * self.X['f_83']
        self.X['mul_5'] = self.X['f_200'] * self.X['f_156']
        self.X['mul_6'] = self.X['f_76'] * self.X['f_156']
        self.X['mul_7'] = self.X['f_76'] * self.X['f_131']
        self.X['mul_8'] = self.X['f_76'] * self.X['f_182']
        self.X['mul_9'] = self.X['f_41'] * self.X['f_182']
        self.X['mul_10'] = self.X['f_11'] * self.X['f_200']

        #
        # self.X['mul'] = self.X['f_84'] * self.X['f_182']

        self.default_columns = [
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

            'mul_0', 'mul_1', 'mul_2', 'mul_3', 'mul_4', 'mul_5', 'mul_6', 'mul_7', 'mul_8', 'mul_9', 'mul_10',

        ]

        self.kf = None
        self.cpu = cpu
        self.pipeline = None

    def get_fold(self, columns, fold_amount=5):
        self.kf = StratifiedKFold(n_splits=fold_amount, shuffle=True)
        self.kf.get_n_splits(self.X[columns], self.Y['target'])

        for train_index, test_index in self.kf.split(self.X[columns], self.Y['target']):
            x_train, x_test = self.X.as_matrix(columns)[train_index], self.X.as_matrix(columns)[test_index]
            y_train, y_test = self.Y.as_matrix()[train_index], self.Y.as_matrix()[test_index]

            return x_train, y_train, x_test, y_test

    def ensemble(self, folds_limit=42):
        answers = []

        pass
        # clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=4,
        #                             n_estimators=1000, n_jobs=self.cpu)
        # clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000,
        #                             n_jobs=self.cpu)
        # clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000,
        #                             n_jobs=self.cpu)
        # clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000,
        #                             n_jobs=self.cpu)

        pass
        # default 0.6742 on seed=42 for full set (search_best_3)
        clf1 = ExtraTreesClassifier(max_features=0.4537270875668709, criterion='entropy', min_samples_leaf=1,
                                    min_samples_split=2, n_estimators=3138, n_jobs=self.cpu)

        pass
        # clf1 = RandomForestClassifier(max_features=0.34808889858456293, criterion='entropy',
        #                               min_samples_split=2, n_estimators=4401, n_jobs=self.cpu)

        pass
        # default
        # clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=2, n_estimators=1000,
        #                             n_jobs=self.cpu)

        self.pipeline = EnsembleVoteClassifier(clfs=[clf1], weights=[1], voting='soft')

        for iteration in range(folds_limit):
            np.random.seed(42 + iteration)

            x_train, y_train, x_test, y_test = self.get_fold(self.default_columns)
            self.pipeline.fit(x_train, y_train)
            preds = self.pipeline.predict(x_test)

            # print(confusion_matrix(y_test, preds))
            matrix_ = confusion_matrix(y_test, preds)
            correct_answers = matrix_[0][0] + matrix_[1][1] + matrix_[2][2] + matrix_[3][3] + matrix_[4][4]
            print('   Correct answers count: ', correct_answers, ' [it: %s]' % iteration)
            answers.append(int(correct_answers))
            if iteration % 5 == 0 and iteration > 0:
                print('Params: mean: %s std: %s best: %s' % (np.mean(answers), np.std(answers), max(answers)))
        print('Params: mean: %s std: %s best: %s' % (np.mean(answers), np.std(answers), max(answers)))

    def answers(self, iter_limit=10):
        self.pipeline.fit(self.X[self.default_columns], self.Y['target'])
        score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=self.names)
        # feature generator block:
        score_dataset['mul_0'] = score_dataset['f_138'] * score_dataset['f_96']
        score_dataset['mul_1'] = score_dataset['f_138'] * score_dataset['f_156']
        score_dataset['mul_2'] = score_dataset['f_11'] * score_dataset['f_200']
        score_dataset['mul_3'] = score_dataset['f_96'] * score_dataset['f_83']
        score_dataset['mul_4'] = score_dataset['f_200'] * score_dataset['f_83']
        score_dataset['mul_5'] = score_dataset['f_200'] * score_dataset['f_156']
        score_dataset['mul_6'] = score_dataset['f_76'] * score_dataset['f_156']
        score_dataset['mul_7'] = score_dataset['f_76'] * score_dataset['f_131']
        score_dataset['mul_8'] = score_dataset['f_76'] * score_dataset['f_182']
        score_dataset['mul_9'] = score_dataset['f_41'] * score_dataset['f_182']
        score_dataset['mul_10'] = score_dataset['f_11'] * score_dataset['f_200']

        predicts = pd.DataFrame()
        for iteration in range(iter_limit):
            if iteration > 0 and iteration % 5 ==0:
                print('[Predict: %s]' % iteration)
            np.random.seed(42 + iteration)
            y_pred = self.pipeline.predict(score_dataset[self.default_columns])
            predicts[iteration] = y_pred

        vote_answer = []
        for pos in range(len(predicts[0])):
            row_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
            for column in predicts.columns:
                row_dict[str(predicts[column].iloc[pos])] += 1
            best_answer_count = 0
            for key in row_dict.keys():
                if row_dict[key] > best_answer_count:
                    best_answer_count = row_dict[key]
                    best_answer = int(key)
            vote_answer.append(best_answer)

        predicts['votes'] = vote_answer
        predicts['diff'] = predicts['votes'] - predicts[0]
        print(predicts[predicts['diff'] != 0])
        print(predicts[predicts['diff'] != 0].shape)

        pd.Series(predicts['votes']).to_csv('data/answer.csv', index=False)


ensemble_class = GetThatEnsemble(cpu=4)
ensemble_class.get_fold(ensemble_class.default_columns)
ensemble_class.ensemble(2)
ensemble_class.answers(1)


"""
Notes:
------------------------
 model: tests:
  ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=2,
                               n_estimators=1000, n_jobs=self.cpu)
  on 32 folds metrics: 444.15625 std: 13.1032185335 best: 475
  after mul search   : 444.96875 std: 11.8756167603 best: 470

  # features:
        self.X['f_138'] * self.X['f_96']  = mean: 446.0 std: 11.5298525576
        self.X['f_138'] * self.X['f_156'] = mean: 446.875 std: 14.0951010993
        self.X['f_11'] * self.X['f_200']  = mean: 446.46875 std: 12.7866736659 best: 478
        self.X['f_96'] * self.X['f_83']   = mean: 445.90625 std: 12.7189803419 best: 478
        self.X['f_200'] * self.X['f_83']  = mean: 446.0625 std: 12.8548081958 best: 481
        self.X['f_200'] * self.X['f_156'] = mean: 446.53125 std: 12.2013943235 best: 481
        self.X['f_76'] * self.X['f_156']  = mean: 445.90625 std: 12.6079324609 best: 478
        self.X['f_76'] * self.X['f_131']  = mean: 445.0 std: 14.2565774294 best: 480
        self.X['f_76'] * self.X['f_182']  = mean: 445.25 std: 13.7431801269 best: 487
        self.X['f_41'] * self.X['f_182']  = mean: 445.3125 std: 14.2728709008 best: 484


------------------------
------------------------
 model: public = 0.6742:
  ExtraTreesClassifier(max_features=0.4537270875668709, criterion='entropy', min_samples_leaf=1, min_samples_split=2,
                       n_estimators=3138, n_jobs=self.cpu)
  on 64 folds metrics: mean: 445.590163934 std: 12.5963526456 best: 481
------------------------
------------------------
model: ensemble public = 0.6756
  clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=4,
                               n_estimators=1000, n_jobs=self.cpu)
  clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000,
                               n_jobs=self.cpu)
  clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000,
                               n_jobs=self.cpu)
  clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000,
                               n_jobs=self.cpu)
  on 23 folds metrics: mean: 445.652173913 std: 14.3302923167 best: 472
------------------------
------------------------
model: ensemble public = 0.6756 ([1, 1, 1, 1]) - forgot about 2 same models in the end
  clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=4,
                               n_estimators=1000, n_jobs=self.cpu)
  clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000,
                               n_jobs=self.cpu)
  clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000,
                               n_jobs=self.cpu)
  clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000,
                               n_jobs=self.cpu)
  on 32 folds metrics: mean: 445.652173913 std: 14.3302923167 best: 472
------------------------
------------------------
model: ensemble public = 0.6756 ([1, 1, 1, 2])
  clf1 = ExtraTreesClassifier(max_features=0.4, min_samples_leaf=1, min_samples_split=4,
                               n_estimators=1000, n_jobs=self.cpu)
  clf2 = ExtraTreesClassifier(criterion="gini", max_features=0.4, min_samples_split=6, n_estimators=1000,
                               n_jobs=self.cpu)
  clf3 = ExtraTreesClassifier(max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=1000,
                               n_jobs=self.cpu)
  clf4 = ExtraTreesClassifier(max_features=0.45, min_samples_leaf=1, min_samples_split=5, n_estimators=1000,
                               n_jobs=self.cpu)
  on 32 folds metrics: mean: 444.4375 std: 13.3532240957 best: 471
------------------------
------------------------
 model: public = 0.6636:
  RandomForestClassifier(max_features=0.34808889858456293, criterion='entropy',
                         min_samples_split=2, n_estimators=4401)
  on 32 folds metrics: 440.5625 std: 12.9757887525 best: 467
------------------------
"""





