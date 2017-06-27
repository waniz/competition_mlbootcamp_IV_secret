import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

names = ['f_' + str(i) for i in range(223)]
X = pd.read_csv('original_data/x_train.csv', delimiter=';', names=names)
Y = pd.read_csv('original_data/y_train.csv', names=['target'], delimiter=';')

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

Y = to_categorical(Y, nb_classes=5)

x_train, x_test, y_train, y_test = train_test_split(X.as_matrix(best_columns), Y, test_size=0.2)

model = Sequential()
model.add(Dense(input_dim=x_train.shape[1], activation='relu', output_dim=11))
model.add(Dense(11, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_json = model.to_json()
with open('models/current_model.json', 'w') as json_file:
    json_file.write(model_json)
filepath = "models/weights_ep_loss_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)
model.fit(x_train, y_train, nb_epoch=200, batch_size=1, verbose=2, callbacks=[reduce_lr])

score, accuracy = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
print('\nAccuracy score :', round(accuracy, 4))
