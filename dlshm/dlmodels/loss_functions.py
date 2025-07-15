
import tensorflow as tf
from keras.src.losses import tversky


#from tensorflow.keras import backend as K

def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights)
    def loss(y_true, y_pred):
        # Apply the softmax activation
        y_pred = tf.nn.softmax(y_pred)
        class_weights32 = tf.cast(class_weights, dtype=tf.float32)

        # Compute the weighted loss
        weights = tf.reduce_sum(class_weights32 * y_true, axis=-1)
        unweighted_loss = tf.reduce_sum(-y_true * tf.math.log(y_pred + 1.0E-7), axis=-1)
        weighted_loss = weights * unweighted_loss
        return tf.reduce_mean(weighted_loss)

    return loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = (numerator + smooth) / (denominator + smooth)
    return 1 - tf.reduce_mean(dice)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    TP = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2,3])
    FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2,3])
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tf.reduce_mean(tversky)


def wrapped_tversky_loss(y_true, y_pred):
    return tf.reduce_mean(tversky(y_true, y_pred, alpha=0.7, beta=0.3, axis=[1, 2, 3]))


def weighted_tversky_loss(class_weights):
    class_weights = tf.constant(class_weights)
    def loss(y_true, y_pred, alpha=0.7, beta=0.7, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        TP = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2])
        FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2])

        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        if class_weights is not None:
            tversky = tversky * tf.constant(class_weights, dtype=tf.float32)

        return 1 - tf.reduce_mean(tversky)

    return loss

def weighted_focal_tversky_loss(class_weights):
    class_weights = tf.constant(class_weights)
    def loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6):
        """
            Focal Tversky Loss with per-class weights.

            Parameters:
            - alpha, beta: Tversky weighting (typically alpha > beta for imbalanced data)
            - gamma: focal exponent (>1 emphasizes hard examples)
            - class_weights: list or array of per-class weights, e.g. [0.0, 0.5, 0.5]
            """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute per-class Tversky index
        TP = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2])
        FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2])

        tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        # Apply focal modulating term
        focal_tversky = tf.pow(1.0 - tversky_index, gamma)

        # Apply class weights if provided
        if class_weights is not None:
            class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
            focal_tversky = focal_tversky * class_weights_tensor

        return tf.reduce_mean(focal_tversky)

    return loss

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    TP = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2,3])
    FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2,3])

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return tf.reduce_mean(tf.pow((1 - tversky), gamma))
