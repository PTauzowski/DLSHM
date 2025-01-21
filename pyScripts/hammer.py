import numpy as np
import tensorflow as tf
import pandas as pd
from keras.src.losses import Huber
from keras.src.regularizers import l2

print(tf.__version__)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras import layers

# Example data
#X = np.random.rand(100, 3)  # 100 samples, each with 3 features
#y = np.random.rand(100, 3)  # 100 samples, each with 3 target values


TASK_PATH = '/Users/piotrek/Documents/Papers/Hammer'
XLSX_FILE = TASK_PATH + '/benedek-eng-diagramok.xlsx'

# Read the Excel file
df = pd.read_excel(XLSX_FILE, sheet_name='Munka1')

# Select the columns you want to read
columns_all = df[['sample #', 'saturation', 'frozen state', 'density', 'Ultrasound velocity (p waves)', 'ultrasound velocity of s waves ', 'poisson', 'E', 'G', 'temperature', 'impact work', 'compressive strenght', 'E/ucs', 'G/ucs', 'p/s', 's/p', 'nű/ucs', 'ró/ucs', 'M', 'ucs/ró', 'avarage', 'deviation (sigma)', '3xdeviation (3sigma)']]
columns_x = df[[ 'density', 'temperature', 'compressive strenght', 'Ultrasound velocity (p waves)', 'ultrasound velocity of s waves ', 'E', 'G' ]].to_numpy()
columns_y = df[[ 'impact work' ]].to_numpy()

row_to_remove = 24  # Remove the second row

# Remove the row
columns_x = matrix = np.delete(columns_x, row_to_remove, axis=0)
columns_y = matrix = np.delete(columns_y, row_to_remove, axis=0)

X_DIM=columns_x.shape[1]
Y_DIM=columns_y.shape[1]
N_SAMPLES=89

# Convert to Numpy array
X = columns_x[0:N_SAMPLES,:]
y = columns_y[0:N_SAMPLES,:]

#print(numpy_array)

#X = np.random.rand(N_SAMPLES, X_DIM)  # 100 samples, each with 3 features
#y = np.random.rand(N_SAMPLES, Y_DIM)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_DIM, activation='linear'))
    model.add(Dense(32, input_dim=X_DIM, activation='linear'))
    model.add(Dense(Y_DIM, activation='linear'))  # Output layer with 3 units (for 3-dimensional output)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model


from sklearn.model_selection import KFold, StratifiedKFold
from scikeras.wrappers import KerasClassifier, KerasRegressor


# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=35)

# # Perform cross-validation
# results = []
# for train_index, test_index in kf.split(X):
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
#
#      model = KerasRegressor(build_fn=create_model, epochs=500, batch_size=10, verbose=0)
#      model.fit(X_train, y_train)
#      scores = model.score(X_test, y_test)
#      results.append(scores)
#
# print(f'Cross-Validation Scores: {results}')
# print(f'Mean Score: {np.mean(results)}')


columns_x = df[[ 'density', 'temperature', 'compressive strenght', 'Ultrasound velocity (p waves)', 'ultrasound velocity of s waves '  ]].to_numpy()
columns_y = df[[ 'impact work' ]].to_numpy()

for k in range(columns_y.shape[0]):
    if (df.iloc[k, 1] == 'dry'):
        columns_y[k]=0
    else:
        columns_y[k]=1


columns_x = matrix = np.delete(columns_x, row_to_remove, axis=0)
columns_y = matrix = np.delete(columns_y, row_to_remove, axis=0)

X = columns_x[0:N_SAMPLES,:]
y = columns_y[0:N_SAMPLES,:].astype(int)

def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_DIM, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Use StratifiedKFold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
results = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Reinitialize the model for each fold
    model = KerasClassifier(build_fn=create_model, epochs=500, batch_size=10, verbose=0)

    # Train and evaluate the model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Accuracy on the test fold
    results.append(score)


# Print results
print("Stratified Cross-Validation Scores:", results)
print("Mean Accuracy:", np.mean(results))