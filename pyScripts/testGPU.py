import tensorflow as tf

import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow can access a GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available.")
    exit(-1)

# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)




import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

# Step 1: Generate a Synthetic Dataset
X, y = make_classification(
    n_samples=1000,       # Number of samples
    n_features=20,        # Number of features
    n_informative=15,     # Number of informative features
    n_redundant=5,        # Number of redundant features
    random_state=42       # Reproducibility
)

# Normalize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Step 2: Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Build the Model
model = Sequential([
    Dense(64, activation='relu', input_dim=X.shape[1]),  # Input layer
    Dense(32, activation='relu'),                        # Hidden layer
    Dense(1, activation='sigmoid')                       # Output layer (binary classification)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),                 # Optimizer
    loss=BinaryCrossentropy(),                           # Loss function
    metrics=[Accuracy()]                                 # Metric for evaluation
)

# Step 4: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,                  # Number of epochs
    batch_size=32,              # Batch size
    validation_split=0.2,       # Use 20% of the training data for validation
    verbose=1                   # Display progress
)

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 6: Make Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Threshold predictions at 0.5
print(f"Predicted Labels:\n{y_pred[:10].flatten()}")
print(f"True Labels:\n{y_test[:10]}")
