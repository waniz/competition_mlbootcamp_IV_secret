import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
score_dataset = pd.read_csv('original_data/x_test.csv', delimiter=';', names=names)

y_pred = [3 for _ in range(len(score_dataset['f_1']))]
pd.Series(y_pred).to_csv('data/answer.csv', index=False)
