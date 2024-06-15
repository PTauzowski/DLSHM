
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K

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
    y = np.zeros((num_samples, output_length,1))

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
        #y[i,:,0] = y_full
        y[i,:,0] = sum(np.sin(frequency * y_full + phase) for frequency, phase in zip(frequencies, phases))

        X[i,:,0]= y_full[perm]
        X[i,:,1]=y[i,perm,0]

        # X[i,:,0]=y[i,perm,0]
        # X[i,:,1]=y[i,perm,1]

    return X, y


class LRFinder:
    def __init__(self, model, stop_factor=4):
        self.model = model
        self.stop_factor = stop_factor
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        loss = logs["loss"]
        lr = K.get_value(self.model.optimizer.lr)
        self.losses.append(loss)
        self.lrs.append(lr)

        if loss < self.best_loss:
            self.best_loss = loss
        if loss > self.best_loss * self.stop_factor:
            self.model.stop_training = True

        K.set_value(self.model.optimizer.lr, lr * 1.1)

    def plot_loss(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.show()


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
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=4)(x)
    x = layers.UpSampling1D(size=2)(x)

    output_layer = layers.Conv1D(2, kernel_size=3, activation='linear', padding='same', dilation_rate=1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse',metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.Accuracy()])

    return model

# Residual block with dilated convolutions
def residual_block(x, dilation_rate):
    shortcut = x
    x = layers.Conv1D(128, kernel_size=3, padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


# Define the model with dilated convolutions and residual connections
def create_dilated_conv_model_with_residuals(input_length=128, output_length=4096):
    input_layer = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(128, kernel_size=3, padding='same', dilation_rate=1)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, dilation_rate=1)
    x = residual_block(x, dilation_rate=2)
    x = residual_block(x, dilation_rate=4)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)

    x = layers.Dense(output_length // 32, activation='relu')(x)
    x = layers.Reshape((output_length // 32, 1))(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=4)(x)
    x = layers.UpSampling1D(size=2)(x)

    output_layer = layers.Conv1D(1, kernel_size=3, activation='linear', padding='same', dilation_rate=1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse',
                  metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.Accuracy()])

    return model


IS_TRAINING_MODE = True
LEARNING_RATE = 0.001
EPOCHS=50
BATCH_SIZE=64
num_samples =5000
input_length = 128
output_length = 4096
X_train, y_train = generate_sinusoidal_data(num_samples, input_length, output_length,3)

sample_index = 20  # Choose a sample to visualize
X_sample = X_train[sample_index,:]
y_true = y_train[sample_index,:]

sample_index2 = 50  # Choose a sample to visualize
X_sample2 = X_train[sample_index2,:]
y_true2 = y_train[sample_index2,:]

# Plot the input signal, true output signal, and predicted output signal
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(y_true[:],  label='True signal (densely sampled)')
#plt.plot(np.linspace(0, 2 * np.pi, input_length), X_sample.flatten(), 'o', label='Input Signal (rarely sampled)')
plt.title('True output signal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true2[:], label='True signal (densely sampled)')
#plt.plot(np.linspace(0, 2 * np.pi, input_length), X_sample.flatten(), 'o', label='Input Signal (rarely sampled)')
plt.title('True output signal')
plt.legend()

plt.tight_layout()
plt.savefig('output_plot.svg', format='svg')
plt.show()

# Train the model
model=None
if IS_TRAINING_MODE:
    model = create_dilated_conv_model_with_residuals(input_length, output_length)
    model.summary()
    lr_finder = LRFinder(model)
    lr_callback = LambdaCallback(on_batch_end=lambda batch, logs: lr_finder.on_batch_end(batch, logs))
    #model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[lr_callback])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)
    model.save('signals_model.h5')
    #lr_finder.plot_loss()
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