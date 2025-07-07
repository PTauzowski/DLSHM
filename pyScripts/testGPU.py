import tensorflow as tf

import tensorflow as tf


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow can access a GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available.")
    exit(-1)


