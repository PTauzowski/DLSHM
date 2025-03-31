import tensorflow as tf
from keras.layers import RandomRotation
import matplotlib.pyplot as plt

# Define the random rotation layer
random_rotation = RandomRotation(factor=0.1)  # Rotate between -72° and +72°

# Sample image and mask (batch of 1)
image = tf.random.uniform(shape=(1, 100, 100, 3))  # Random image
mask = tf.random.uniform(shape=(1, 100, 100, 8), minval=0, maxval=8, dtype=tf.int32)  # Random mask with 8 classes

# Apply the same rotation to both the image and mask
augmented_image = random_rotation(image)
augmented_mask = random_rotation(tf.cast(mask, tf.float32))  # Cast mask to float before applying transformation
augmented_mask = tf.cast(augmented_mask, tf.int32)  # Convert back to int

# Visualize the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(image[0].numpy())  # Original image
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(augmented_image[0].numpy())  # Augmented image
ax[1].set_title("Augmented Image")
ax[1].axis("off")

plt.show()

# Visualize the original and augmented masks (first channel)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(mask[0, :, :, 0].numpy(), cmap="gray")  # Original mask
ax[0].set_title("Original Mask")
ax[0].axis("off")

ax[1].imshow(augmented_mask[0, :, :, 0].numpy(), cmap="gray")  # Augmented mask
ax[1].set_title("Augmented Mask")
ax[1].axis("off")

plt.show()
