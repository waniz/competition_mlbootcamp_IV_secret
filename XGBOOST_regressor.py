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

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

print('Class :',
      Y[Y['target'] == 0].shape, Y[Y['target'] == 1].shape, Y[Y['target'] == 2].shape,
      Y[Y['target'] == 3].shape, Y[Y['target'] == 4].shape)
print('Train set:', X.shape, Y.shape)

model_test = xgb.XGBRegressor()
model_test.fit(X.as_matrix(), Y.as_matrix(), verbose=True)
y_pred = model_test.predict(X.as_matrix())
print(y_pred)

X['xgb_regressor'] = y_pred
X.to_csv('data/x_train_regressor_0.csv', index=False)


score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)
y_pred = model_test.predict(score_dataset.as_matrix())
score_dataset['xgb_regressor'] = y_pred
score_dataset.to_csv('data/y_train_regressor_0.csv', index=False)




