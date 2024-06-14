
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def generate_sinusoidal_data(num_samples, input_length, output_length, num_sinusoids=3):
    X = np.zeros((num_samples, input_length))
    y = np.zeros((num_samples, output_length))

    min_frequency = 90
    max_frequency = 800

    for i in range(num_samples):
        num_components = np.random.randint(1, num_sinusoids + 1)
        frequencies = np.random.uniform(min_frequency, max_frequency, num_components)
        phases = np.random.uniform(0, 2 * np.pi, num_components)

        x = np.linspace(0, 2 * np.pi, input_length)
        y_full = np.linspace(0, 2 * np.pi, output_length)

        #X[i] = sum(np.sin(frequency * x + phase) for frequency, phase in zip(frequencies, phases))
        y[i] = sum(np.sin(frequency * y_full + phase) for frequency, phase in zip(frequencies, phases))

        permns = np.round(np.random.rand(input_length) * (output_length-1)).astype(int)  # Ensure integer indices
        perm = np.sort(permns)
        X[i]=y[i,perm]

    return X, y


def generate_sinusoidal_data_with_time(num_samples, input_length, output_length, num_sinusoids=3):
    X = np.zeros((num_samples, input_length,2))
    y = np.zeros((num_samples, output_length,2))

    min_frequency = 90
    max_frequency = 800

    for i in range(num_samples):
        num_components = np.random.randint(1, num_sinusoids + 1)
        frequencies = np.random.uniform(min_frequency, max_frequency, num_components)
        phases = np.random.uniform(0, 2 * np.pi, num_components)
        permns = np.round(np.random.rand(input_length) * (output_length-1)).astype(int)  # Ensure integer indices
        perm = np.sort(permns)

        x = np.linspace(0, 2 * np.pi, input_length)
        y_full = np.linspace(0, 2 * np.pi, output_length)

        #X[i] = sum(np.sin(frequency * x + phase) for frequency, phase in zip(frequencies, phases))
        y[i,:,0] = y_full
        y[i,:,1] = sum(np.sin(frequency * y_full + phase) for frequency, phase in zip(frequencies, phases))

        X[i,:,0]=y[i,perm,0]
        X[i,:,1]=y[i,perm,1]

    return X, y

def create_model(input_length=128, output_length=4096):
    model = Sequential()

    # Encoder
    model.add(layers.Reshape((input_length, 1), input_shape=(input_length,1)))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    # Decoder
    model.add(layers.Dense(output_length // 32, activation='relu'))
    model.add(layers.Reshape((output_length // 32, 1)))
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='linear', padding='same'))
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(1, kernel_size=3, activation='linear', padding='same'))

    #model.add(layers.Reshape((output_length,1)))
    model.add(layers.Reshape((output_length, 1)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse',metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.Accuracy()])

    return model


def create_dilated_conv_model_with_time(input_length=128, output_length=4096):
    input_layer = layers.Input(shape=(input_length, 2))

    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(input_layer)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same', dilation_rate=4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)

    x = layers.Dense(output_length // 32, activation='relu')(x)
    x = layers.Reshape((output_length // 32, 1))(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=4)(x)
    x = layers.UpSampling1D(size=2)(x)

    output_layer = layers.Conv1D(2, kernel_size=3, activation='linear', padding='same', dilation_rate=1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse',metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.Accuracy()])

    return model

IS_TRAINING_MODE = False
LEARNING_RATE = 0.001
num_samples =20000
input_length = 128
output_length = 4096
X_train, y_train = generate_sinusoidal_data_with_time(num_samples, input_length, output_length,3)

sample_index = 20  # Choose a sample to visualize
X_sample = X_train[sample_index,:,:]
y_true = y_train[sample_index,:,:]

sample_index2 = 50  # Choose a sample to visualize
X_sample2 = X_train[sample_index2,:,:]
y_true2 = y_train[sample_index2,:,:]

# Plot the input signal, true output signal, and predicted output signal
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_true[:,0],y_true[:,1],  label='True signal (densely sampled)')
plt.plot(X_sample[:,0], X_sample[:,1], 'o', label='True Output Signal (densely sampled)')
#plt.plot(np.linspace(0, 2 * np.pi, input_length), X_sample.flatten(), 'o', label='Input Signal (rarely sampled)')
plt.title('True output signal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true2[:,0],y_true2[:,1], label='True signal (densely sampled)')
plt.plot(X_sample2[:,0], X_sample2[:,1], 'o', label='True Output Signal (densely sampled)')
#plt.plot(np.linspace(0, 2 * np.pi, input_length), X_sample.flatten(), 'o', label='Input Signal (rarely sampled)')
plt.title('True output signal')
plt.legend()

plt.tight_layout()
plt.savefig('output_plot.svg', format='svg')
plt.show()

# Train the model
model=None
if IS_TRAINING_MODE:
    model = create_dilated_conv_model_with_time(input_length, output_length)
    model.summary()
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)
    model.save('signals_model.h5')
else:
    model = tf.keras.models.load_model('signals_model.h5')

# Predict using the trained model
X_sample_test, y_true_test = generate_sinusoidal_data(num_samples, input_length, output_length,3)

sample_index = 20  # Choose a sample to visualize
X_sample = X_sample_test[sample_index,:,1]
y_true = y_true_test[sample_index,:,1]

y_pred = model.predict(X_sample).flatten()

# Plot the input signal, true output signal, and predicted output signal
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_true_test[sample_index,:,0], y_true, label='True Output Signal (densely sampled)')
plt.plot(X_sample_test[sample_index,:,0], X_sample, 'o', label='Input Signal (rarely sampled)')
plt.title('True output signal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true_test[sample_index,:,0], y_true, label='True signal (densely sampled)')
plt.plot(y_pred[sample_index,:,0], y_pred[sample_index,:,1], label='Predicted output signal (densely sampled)')
plt.title('Predicted + true output signal')
plt.legend()

plt.tight_layout()
plt.savefig('output_plot.svg', format='svg')
plt.show()