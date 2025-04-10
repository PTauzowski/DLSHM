
import os

from dlshm.dlmodels.custom_models import DeeplabV3Plus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"


import tensorflow as tf

import tensorflow as tf
print(tf.__version__)
print(tf.keras)

from tensorflow.keras import layers, models

from dlshm.dlimages.data_processing import ICSHM_DMG_Converter
from dlshm.dlmodels.c_unet import custom_unet


import tensorflow as tf
import segmentation_models as sm
from dlshm.dlimages.ICSHM_tasks import ICSHM_structural_task, ICSHM_damage_task, multi_augmentation_training_structural

# available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
#                   'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121',
#                   'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
#                   'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

RES_X=640
RES_Y=320
BATCH_SIZE=16
TASK_PATH = '/home/piotrek/Computations/Ai/ICSHM'
SOURCE_PATH = '/home/piotrek/Computations/Ai/Data/Tokaido_dataset_share'

# TASK_NAME='ICSHM_STRUCT_UNET_rn18'
# create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# #create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=24, num_classes=4, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_UNET_rn18'
# create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# #create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=24, num_classes=3, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )




#TASK_NAME='ICSHM_STRUCT_UNET_rn101'
# create_unet_fn = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_UNET_rn101'
# create_unet_fn = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_inceptionv3'
# create_unet_fn = lambda: sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_UNET_inceptionv3'
# create_unet_fn = lambda: sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_STRUCT_CUSTOM_UNET'
create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=16, num_classes=4, output_activation="softmax")
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_DMG_CUSTOM_UNET'
create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=16, num_classes=3, output_activation="softmax")
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_np'
create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=False)
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_DMG_DEEPLABV3p_np'
create_model_fn = lambda: DeeplabV3Plus((RES_Y, RES_X, 3), 3, output_activation="softmax",is_pretrained=False)
create_dmg_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_dmg_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_STRUCT_DEEPLABV3p'
create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=True)
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE  )


TASK_NAME='ICSHM_DMG_DEEPLABV3p'
create_model_fn = lambda: DeeplabV3Plus((RES_Y, RES_X, 3), 3, output_activation="softmax",is_pretrained=True)
create_dmg_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_dmg_task_fn, BATCH_SIZE  )
