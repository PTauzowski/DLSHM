import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

I1 = plt.imread('testIm1.jpg')
I2 = tf.image.random_brightness(I1, max_delta=0.2).numpy()
plt.figure()
plt.imshow([I1, I2])
plt.show()