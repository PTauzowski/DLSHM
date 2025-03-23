import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


def test_rgb_postprocess(filename, x, y, result):
    fig, axp = plt.subplots(N_CLASSES, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 8):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i] > 0.5)
        axp[i, 3].imshow(result[0, :, :, i])
    plt.savefig(filename)
    plt.close(fig)

def test_dmg_segmentation(pathname, x, y, result):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    source_name=os.path.join(path,name+"_source")+extension
    test_name = os.path.join(path,name + "_result")+extension
    # Define the color palette for the segmentation masks
    colors = np.array([
         [0, 0, 0],  # background
         [1, 0, 0],  # mask 1 (red)
         [0, 1, 0],  # mask 2 (green)
         [0, 0, 1]  # mask 3 (blue)
    ], dtype=np.float32)

    accuracy =  np.mean( y == (result > 0.6).astype(int) )
    epsilon = 1.0E-07
    y_pred = tf.clip_by_value(result, clip_value_min=epsilon, clip_value_max=1.0 - epsilon)
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=-1))

    nmasks = y.shape[3]
    masks = colors[np.argmax(result, axis=-1)]
    sourse_masks = colors[np.argmax(y, axis=-1)]

    alpha = 0.6
    blended = cv.addWeighted(masks[0,], 1-alpha, x[0,], alpha, 0)
    source_blended = cv.addWeighted(sourse_masks[0,], 1 - alpha, x[0,], alpha, 0)
    # Display the result in a window
    cv.imwrite(source_name, source_blended*255)
    cv.imwrite(test_name,blended*255)
    #print(name," accuracy = ",accuracy," loss = ", loss, "\n")

def write_prediction_segmentated(pathname, x, y):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    source_name=os.path.join(path,name+"_source")+extension
    test_name = os.path.join(path,name + "_result")+extension
    # Define the color palette for the segmentation masks
    colors = np.array([
         [0, 0, 0],  # background
         [1, 0, 0],  # mask 1 (red)
         [0, 1, 0],  # mask 2 (green)
         [0, 0, 1]  # mask 3 (blue)
    ], dtype=np.float32)

    nmasks = y.shape[2]
    masks = y #colors[np.argmax(y, axis=-1)]

    alpha = 0.3
    blended = cv.addWeighted(masks, 1-alpha, x, alpha, 0)

    # Display the result in a window
    cv.imwrite(source_name, x*255)
    cv.imwrite(test_name,blended*255)


def write_prediction_segmentated2(pathname, x, y):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    source_name=os.path.join(path,name+"_source")+extension
    test_name = os.path.join(path,name + "_result")+extension
    # Define the color palette for the segmentation masks
    colors = np.array([
         [0, 0, 0],  # background
         [1, 0, 0],  # mask 1 (red)
         [0, 1, 0],  # mask 2 (green)
         [0, 0, 1]  # mask 3 (blue)
    ], dtype=np.float32)

    nmasks = y.shape[2]
    masks = colors[np.argmax(y, axis=-1)]

    alpha = 0.6
    blended = cv.addWeighted(masks, 1-alpha, x, alpha, 0)

    # Display the result in a window
    cv.imwrite(source_name, x*255)
    cv.imwrite(test_name,blended*255)

def write_smooth_masks(pathname, x, y):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    source_name = os.path.join(path, name + "_source") + extension
    test_name = os.path.join(path, name + "_result") + extension
    class_index = 0  # Class to visualize
    prob_map = y[:, :, class_index]

    cv.imwrite(source_name, (x * 255).astype(np.uint8))

    # Plot the heatmap
    plt.imshow(prob_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Probability Heatmap for Class {class_index}")

    # Save the heatmap to a file
    plt.savefig(test_name)  # Save as PNG file
    plt.close()  # Close t



