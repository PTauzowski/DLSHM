import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def augment_random_image_transformation(X, Y):

    """Random flipping of the image (both vertical and horisontal) taking care on the seed - mask remains consistent with corresponding training image"""

    seed = 42
    rnI = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    if rnI == 1:
        Xout = tf.image.flip_left_right(X)
        Yout = tf.image.flip_left_right(Y)
    elif rnI == 2:
        Xout = tf.image.flip_up_down(X)
        Yout = tf.image.flip_up_down(Y)
    elif rnI > 2:
        angle = np.random.uniform(-30, 30)
        (Xh, Xw) = X.shape[:2]
        Xcenter = (Xw // 2, Xh // 2)
        (Yh, Yw) = Y.shape[:2]
        Ycenter = (Yw // 2, Yh // 2)
        MX = cv2.getRotationMatrix2D(Xcenter, np.int32(angle), 1.0)
        MY = cv2.getRotationMatrix2D(Ycenter, np.int32(angle), 1.0)
        Xout = cv2.warpAffine(X, MX, (Xw, Xh))
        Yout = cv2.warpAffine(Y, MY, (Yw, Yh))
    return Xout, Yout

I1 = plt.imread('testIm1.jpg')
I2 = plt.imread('testIm2.jpg')
# I3 = tf.image.rotate(I1, 1, interpolation = 'bilinear', fill_mode='reflect')
I3, I4 = augment_random_image_transformation(I1, I2)
f, axarr = plt.subplots(2,2)
axarr[0, 0].imshow(I1)
axarr[0, 1].imshow(I2)
axarr[1, 0].imshow(I3)
axarr[1, 1].imshow(I4)
plt.show()