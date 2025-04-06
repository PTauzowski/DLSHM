import gc
import os

import tensorflow as tf
from tensorflow.keras import layers, models

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras layers: {layers.Dense}")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
import segmentation_models as sm
from dlshm.dlimages.ICSHM_tasks import ICSHM_structural_task, multi_augmentation_training_structural

# available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
#                   'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121',
#                   'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
#                   'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']
RES_X=640
RES_Y=320
BATCH_SIZE=32
TASK_PATH = '/home/piotrek/Computations/Ai/ICSHM'
SOURCE_PATH = '/home/piotrek/Computations/Ai/Data/Tokaido_dataset_share'

TASK_NAME='ICSHM_STRUCT_UNET_rn18'
create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )

