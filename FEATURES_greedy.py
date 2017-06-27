import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import warnings


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

good_features_drop_no_corr = [
    'f_0', 'f_1', 'f_2',
    'f_3',
    'f_5', 'f_6', 'f_7', 'f_8', 'f_10',
    'f_11',
    'f_12', 'f_13',
    'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20',
    'f_21',
    'f_22', 'f_23', 'f_26', 'f_27', 'f_28',
    'f_29', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_41', 'f_42',
    'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_51', 'f_52', 'f_54', 'f_55',
    'f_57', 'f_58', 'f_59', 'f_60', 'f_61', 'f_62', 'f_63', 'f_64', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70',
    'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_81', 'f_82', 'f_83', 'f_84',
    'f_85', 'f_87', 'f_88', 'f_89', 'f_90', 'f_91', 'f_92', 'f_93', 'f_94', 'f_95', 'f_96', 'f_97', 'f_98',
    'f_99', 'f_101', 'f_102', 'f_103', 'f_104', 'f_105', 'f_106', 'f_107', 'f_108', 'f_109', 'f_110', 'f_111',
    'f_112', 'f_113', 'f_114', 'f_115', 'f_116', 'f_117', 'f_118', 'f_119', 'f_120', 'f_121', 'f_122', 'f_123', 'f_124',
    'f_125', 'f_126', 'f_127', 'f_128', 'f_129', 'f_130', 'f_131', 'f_133', 'f_134', 'f_135', 'f_136', 'f_137',
    'f_138', 'f_139', 'f_140', 'f_141', 'f_142', 'f_143', 'f_144', 'f_146', 'f_147', 'f_148', 'f_150',
    'f_152', 'f_153', 'f_154', 'f_155', 'f_156', 'f_157', 'f_158', 'f_160', 'f_162', 'f_163',
    'f_164', 'f_165', 'f_167', 'f_168', 'f_169', 'f_170', 'f_171', 'f_172', 'f_174', 'f_175', 'f_176',
    'f_177', 'f_178', 'f_179', 'f_180', 'f_181', 'f_182', 'f_183', 'f_184', 'f_185', 'f_186', 'f_187', 'f_188',
    'f_190', 'f_191', 'f_193', 'f_195', 'f_198', 'f_199', 'f_200', 'f_201',
    'f_204', 'f_205', 'f_206', 'f_208', 'f_209', 'f_210', 'f_211', 'f_212', 'f_213', 'f_214', 'f_215',
    'f_216', 'f_217', 'f_218', 'f_219', 'f_220', 'f_221', 'f_222'
]

params = {
    'objective': 'multi:softmax',
    'eta': 0.016,
    'silent': 1,
    'nthread': 1,
    'eval_metric': ['merror', 'mlogloss'],
    'num_class': 5,
}

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

# best_features = []
# best_feature, min_metrics = [], 100
# for one_feature in good_features_drop_no_corr:
#     features = [one_feature]
#     d_train = xgb.DMatrix(X[features], label=Y)
#     model = xgb.cv(params, d_train, 1200, nfold=5, stratified=True, early_stopping_rounds=300)
#     print('Features" %s, mlogloss: %s' % (features, min(model['test-mlogloss-mean'])))
#     if min_metrics >= min(model['test-mlogloss-mean']):
#         best_feature = [one_feature]
#         min_metrics = min(model['test-mlogloss-mean'])
#
# print('Best feature in round: %s, %s', (best_feature, min_metrics))
# best_features.append(best_feature)

# best_features = [
#     'f_138', 'f_11', 'f_96', 'f_200', 'f_32', 'f_76', 'f_79', 'f_41', 'f_83', 'f_156', 'f_131', 'f_147', 'f_187'
# ]

best_features = [
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

super_mlogloss, super_best_params = [], []
for iteration in range(len(best_features), 50):
    best_feature, min_metrics = [], 100
    for one_feature in good_features_drop_no_corr:
        features = best_features[:]
        if one_feature in best_features:
            continue
        features.append(one_feature)

        d_train = xgb.DMatrix(X[features], label=Y)
        model = xgb.cv(params, d_train, 2000, nfold=5, stratified=True, early_stopping_rounds=300)

        print('Features %s %s, mlogloss: %s' % (iteration, features, min(model['test-mlogloss-mean'])))
        if min_metrics >= min(model['test-mlogloss-mean']):
            best_feature = one_feature
            min_metrics = min(model['test-mlogloss-mean'])

    print('Best feature in round %s: %s, %s' % (iteration, best_feature, min_metrics))
    best_features.append(best_feature)

    super_mlogloss.append(min_metrics)
    super_best_params.append(best_feature)

    results = pd.DataFrame()
    results['mlogloss'] = super_mlogloss
    results['features'] = super_best_params
    results.to_csv('results/features_set.csv', index=False)

    print('Current best: ', best_features)
