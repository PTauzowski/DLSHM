import numpy as np
import tensorflow as tf
import pandas as pd
from keras.layers import Dropout
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
#from keras.wrappers.scikit_learn import KerasClassifier

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
    model.add(Dense(32, activation='linear'))
    model.add(Dense(Y_DIM, activation='linear'))  # Output layer with 3 units (for 3-dimensional output)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model


from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

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

from sklearn.datasets import make_classification
# Step 1: Generate a Synthetic Dataset
X, y = make_classification(
    n_samples=1000,       # Number of samples
    n_features=20,        # Number of features
    n_informative=15,     # Number of informative features
    n_redundant=5,        # Number of redundant features
    random_state=42       # Reproducibility
)


columns_x = df[[ 'density',   'impact work'  ]].to_numpy()
columns_y = df[[ 'impact work' ]].to_numpy()

columns_x  = np.delete(columns_x, row_to_remove, axis=0)
columns_y  = np.delete(columns_y, row_to_remove, axis=0)

y=np.zeros((N_SAMPLES),dtype=np.int32)

for k in range(N_SAMPLES):
    if (df.iloc[k, 1] == 'dry'):
        y[k]=0
    elif (df.iloc[k, 1] == 'saturated'):
        y[k]=1

X = columns_x[0:N_SAMPLES,:].astype(np.float32)

y = y.ravel()

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(type(y))
print(np.unique(y))

def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

model = Sequential([
    Dense(64, activation='relu', input_dim=X.shape[1]),  # Input layer
    Dense(32, activation='relu'),                        # Hidden layer
    Dense(1, activation='sigmoid')                       # Output layer (binary classification)
])

model = create_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),                 # Optimizer
    loss=BinaryCrossentropy(),                           # Loss function
    metrics=[Accuracy()]                                 # Metric for evaluation
)

# Step 4: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=100,                  # Number of epochs
    batch_size=64,              # Batch size
    validation_split=0.1,       # Use 20% of the training data for validation
    verbose=1                   # Display progress
)

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 6: Make Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Threshold predictions at 0.5
print(f"Predicted Labels:\n{y_pred[:40].flatten()}")
print(f"True Labels:\n{y_test[:40]}")

import matplotlib.pyplot as plt

# Step 6: Plot Training History
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Use StratifiedKFold
# skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# results = []
#
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # Reinitialize the model for each fold
#     model = KerasClassifier(build_fn=create_model, epochs=500, batch_size=10, verbose=0)
#
#     # Train and evaluate the model
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)  # Accuracy on the test fold
#     results.append(score)
#
#
# # Print results
# print("Stratified Cross-Validation Scores:", results)
# print("Mean Accuracy:", np.mean(results))