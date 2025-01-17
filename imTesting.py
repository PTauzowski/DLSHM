import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import datetime as dt

def augment_random_image_quality(X):

    """Random decreasing the quality of the image for the data augmentation purposes - applied ony to the image (withut affecting the corresponding mask)"""

    rnI = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    rnI = 3
    if rnI == 0:
        Xout = X
    elif rnI == 1:
        Xout = tf.image.random_brightness(X, max_delta=0.8).numpy()       # Random brightness
    elif rnI == 2:
        Xout = tf.image.random_contrast(X, lower=0.1, upper=1.9).numpy()  # Random contrast
    elif rnI == 3:
        noiseIndic = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.int32)
        if noiseIndic == 0:
            noise = tf.random.normal(shape=tf.shape(X), mean=0, stddev=50, dtype=tf.float32)
            X = X + noise
        else:
            salt = tf.cast(tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < 50, tf.float32)
            pepper =  tf.cast(tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < 50, tf.float32) # salt - pepper
            X = X + salt - pepper
        Xout = tf.clip_by_value(X, 0.0, 1.0).numpy()  # Ensure pixel values are in [0, 1]

    return Xout

I1 = plt.imread('testIm1.jpg')
I2 = plt.imread('testIm2.jpg')
# I3 = tf.image.rotate(I1, 1, interpolation = 'bilinear', fill_mode='reflect')
t0 = dt.datetime.now()
I4 = augment_random_image_quality(I2)
print("execution time:", dt.datetime.now() - t0, " s")

f, axarr = plt.subplots(2,2)
axarr[0, 0].imshow(I1)
axarr[0, 1].imshow(I2)
axarr[1, 0].imshow(I4)
axarr[1, 1].imshow(I4)
plt.show()