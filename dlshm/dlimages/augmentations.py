import tensorflow as tf
import cv2 as cv
import numpy as np


def augment_random_image_transformation(X, Y):

    """Random flipping of the image (both vertical and horisontal) taking care on the seed - mask remains consistent with corresponding training image"""

    rnI = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    if rnI == 0:
        Xout = X
        Yout = Y
    elif rnI == 1:
        Xout = tf.image.flip_left_right(X).numpy()
        Yout = tf.image.flip_left_right(Y).numpy()
    elif rnI == 2:
        Xout = tf.image.flip_up_down(X).numpy()
        Yout = tf.image.flip_up_down(Y).numpy()
    elif rnI == 3:
        angle = np.random.uniform(-30, 30)
        (Xh, Xw) = X.shape[:2]
        Xcenter = (Xw // 2, Xh // 2)
        (Yh, Yw) = Y.shape[:2]
        Ycenter = (Yw // 2, Yh // 2)
        MX = cv.getRotationMatrix2D(Xcenter, np.int32(angle), 1.0)
        MY = cv.getRotationMatrix2D(Ycenter, np.int32(angle), 1.0)
        Xout = cv.warpAffine(X, MX, (Xw, Xh)).numpy()
        Yout = cv.warpAffine(Y, MY, (Yw, Yh)).numpy()
    return Xout, Yout

def augment_random_image_quality(X,Y):

    """Random decreasing the quality of the image for the data augmentation purposes - applied ony to the image (withut affecting the corresponding mask)"""

    rnI = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    if rnI == 0:
        Xout = X
    elif rnI == 1:
        Xout = tf.image.random_brightness(X, max_delta=0.8).numpy()       # Random brightness
    elif rnI == 2:
        Xout = tf.image.random_contrast(X, lower=0.1, upper=1.9).numpy()  # Random contrast
    elif rnI == 3:
        noiseIndic = f.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.int32)
        if noiseIndic == 0:
            noise = tf.random.normal(shape=tf.shape(X), mean=0, stddev=50, dtype=tf.float32)
        else:
            noise = tf.cast(tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1) < salt_prob, tf.float32) - tf.cast \
                (tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1) < pepper_prob, tf.float32) # salt - pepper
        X = tf.add(X, noise)
        Xout = tf.clip_by_value(X, 0.0, 1.0).numpy()  # Ensure pixel values are in [0, 1]
    return Xout, Y

def augment_photo(X, Y):
    X = tf.image.random_brightness(X, max_delta=0.2).numpy()  # Random brightness
    X = tf.image.random_contrast(X, lower=0.9, upper=1.1).numpy()  # Random contrast
    # Apply a random flip with 50% probability
    if tf.random.uniform([]) > 0.5:  # 50% probability
        X = tf.image.flip_left_right(X)
        Y = tf.image.flip_left_right(Y)
    return X, Y
