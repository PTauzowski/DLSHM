
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv1D, UpSampling1D
import matplotlib.pyplot as plt

def generate_sinusoidal_data(num_samples, input_length, output_length, num_sinusoids=3):
    X = np.zeros((num_samples, input_length))
    y = np.zeros((num_samples, output_length))

    for i in range(num_samples):
        # Create a random number of sinusoids (1 to num_sinusoids)
        num_components = np.random.randint(1, num_sinusoids + 1)

        # Random frequencies and phases
        frequencies = np.random.uniform(0.1, 10.0, num_components)
        phases = np.random.uniform(0, 2 * np.pi, num_components)

        # Generate input and output signals
        x = np.linspace(0, 2 * np.pi, input_length)
        y_full = np.linspace(0, 2 * np.pi, output_length)

        X[i] = sum(np.sin(frequency * x + phase) for frequency, phase in zip(frequencies, phases))
        y[i] = sum(np.sin(frequency * y_full + phase) for frequency, phase in zip(frequencies, phases))

    return X, y

def create_model(input_length=128, output_length=4096):
    model = Sequential()

    # Encoder
    model.add(Reshape((input_length, 1), input_shape=(input_length,)))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    # Decoder
    model.add(Dense(output_length // 32, activation='relu'))
    model.add(Reshape((output_length // 32, 1)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(1, kernel_size=3, activation='linear', padding='same'))

    model.add(Reshape((output_length,)))

    model.compile(optimizer='adam', loss='mse')

    return model

num_samples = 1000
input_length = 128
output_length = 4096
X_train, y_train = generate_sinusoidal_data(num_samples, input_length, output_length)
model = create_model(input_length, output_length)
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


# Predict using the trained model
sample_index = 0  # Choose a sample to visualize
X_sample = X_train[sample_index].reshape(1, -1)
y_true = y_train[sample_index]
y_pred = model.predict(X_sample).flatten()

# Plot the input signal, true output signal, and predicted output signal
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(X_sample.flatten(), label='Input Signal')
plt.title('Input Signal')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(y_true, label='True Output Signal')
plt.title('True Output Signal')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(y_pred, label='Predicted Output Signal')
plt.title('Predicted Output Signal')
plt.legend()

plt.tight_layout()
plt.show()