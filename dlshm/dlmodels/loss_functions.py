
import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights)
    def loss(y_true, y_pred):
        # Apply the softmax activation
        y_pred = tf.nn.softmax(y_pred)
        class_weights32 = tf.cast(class_weights, dtype=tf.float32)

        # Compute the weighted loss
        weights = tf.reduce_sum(class_weights32 * y_true, axis=-1)
        unweighted_loss = tf.reduce_sum(-y_true * tf.math.log(y_pred + K.epsilon()), axis=-1)
        weighted_loss = weights * unweighted_loss
        return tf.reduce_mean(weighted_loss)

    return loss