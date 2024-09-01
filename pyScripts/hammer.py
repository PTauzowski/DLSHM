import numpy as np
import tensorflow as tf

# Example data
X = np.random.rand(100, 3)  # 100 samples, each with 3 features
y = np.random.rand(100, 3)  # 100 samples, each with 3 target values

from tf.keras.models import Sequential
from tf.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=3, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='linear'))  # Output layer with 3 units (for 3-dimensional output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Wrap the Keras model
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=10, verbose=0)

# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Perform cross-validation
results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    scores = model.score(X_test, y_test)
    results.append(scores)

print(f'Cross-Validation Scores: {results}')
print(f'Mean Score: {np.mean(results)}')
