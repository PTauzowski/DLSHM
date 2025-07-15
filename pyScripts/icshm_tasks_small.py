import glob
import os

import keras_hub
from keras import Model
from keras.src.layers import Activation, Conv2D
from keras_hub.src.models.deeplab_v3 import DeepLabV3Backbone

import tensorflow_hub as hub

#from dlshm.dlmodels.transformers import create_vit_model

vit_model_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"  # example feature extractor
vit_encoder = hub.KerasLayer(vit_model_url, trainable=True)

#from transformers import TFSegformerForSemanticSegmentation

from dlshm.dlimages.augmentations import augment_brightness, augment_flip, augment_contrast, augment_gamma, augment_noise, augment_all
from dlshm.dlmodels.basnet import BASNet
from dlshm.dlmodels.custom_models import DeeplabV3Plus, create_deeplab_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"


import tensorflow as tf

import tensorflow as tf
print(tf.__version__)
print(tf.keras)

from tensorflow.keras import layers, models

from dlshm.dlimages.data_processing import ICSHM_DMG_Converter, ICSHM_RGB_Converter, ICSHMDataManager
from dlshm.dlmodels.c_unet import custom_unet
from dlshm.dlresults.postprocess import prepare_excel_multiaugmented_results


import tensorflow as tf
import segmentation_models as sm
from dlshm.dlimages.ICSHM_tasks import ICSHM_structural_task, ICSHM_damage_task, multi_augmentation_training_structural, \
    multi_augmentation_transfer_learning, ICSHM_structural_depth_task

# available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
#                   'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121',
#                   'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
#                   'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

RES_X=320
RES_Y=160
BATCH_SIZE=32
TASK_PATH = '/Users/piotrek/Computations/Ai/ICSHM'
SOURCE_PATH = '/Users/piotrek/Computations/Ai/Data/Tokaido_dataset_share'


augmentations =  (  ("none", "_none", None),
                    ("brightness", "_br", augment_brightness),
                    ("contrast", "_cn", augment_contrast),
                    ("gamma", "_gm", augment_gamma),
                    ("noise", "_ns", augment_noise),
                    ("flip", "_fl", augment_flip),
                    ("rotation", "_rot", augment_flip),
                    ("cutmix", "_cut", augment_flip),
                    ("all", "_all", augment_all))

augmentations_all =  (("all", "_all", augment_all),)

# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_CUSTOM_UNET', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_DEEPLABV3p_LR45', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_DEEPLABV3p_np', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_inceptionv3', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_rn101', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_rn101_lr45', augmentations, nrows=5)
#
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_CUSTOM_UNET', augmentations,nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_DEEPLABV3p_np', augmentations,nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_inceptionv3', augmentations,nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_rn101', augmentations,nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_rn101_lr45', augmentations,nrows=4)

#prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_rn18_small', augmentations, nrows=5)

import keras
keras.config.disable_traceback_filtering()

# TASK_NAME='ICSHM_STRUCTD_UNET_rn101_small'
# #create_unet_fn = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, 4), encoder_weights="imagenet", classes=4, activation="softmax")
# create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,4), num_layers=16, filters=32, num_classes=4, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_depth_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTDsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )

# TASK_NAME='ICSHM_STRUCT_VIT__small'
# create_unet_fn = lambda: create_vit_model( input_shape=(RES_Y, RES_X, 3), num_classes=4 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
#
#TASK_NAME='ICSHM_DMG_UNET_rn18_small'
# create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# #create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=24, num_classes=3, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  , augmentations=augmentations)
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)
#
#
#TASK_NAME='ICSHM_STRUCT_UNET_rn101_small'
# create_unet_fn = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)

#
TASK_NAME='ICSHM_DMG_UNET_rn101_small_tversky'
create_unet_fn = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)

#TASK_NAME='ICSHM_STRUCT_UNET_rn152_small'
# create_unet_fn = lambda: sm.Unet("resnet152", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)

#
#TASK_NAME='ICSHM_DMG_UNET_rn152_small'
# create_unet_fn = lambda: sm.Unet("resnet152", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)

# TASK_NAME='ICSHM_STRUCT_UNET_efnb4_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb4_small_55'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb5_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb5_small_55'
# create_unet_fn = lambda: sm.Unet("efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb6_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb6_small_55'
# create_unet_fn = lambda: sm.Unet("efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_efnb4_small_4'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb4_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb4_small_5'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_small_4'
# create_unet_fn = lambda: sm.Unet("efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_small_5'
# create_unet_fn = lambda: sm.Unet("efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_small_4'
# create_unet_fn = lambda: sm.Unet("efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_small_45'
# create_unet_fn = lambda: sm.Unet("efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_small_5'
# create_unet_fn = lambda: sm.Unet("efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)


#TASK_NAME='ICSHM_DMG_UNET_efnb4_small'
# create_unet_fn = lambda: sm.Unet("efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS,LEARNING_RATE=0.00005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)

#TASK_NAME='ICSHM_STRUCT_UNET_srn101_small'
# create_unet_fn = lambda: sm.Unet("seresnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS,LEARNING_RATE=0.00005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)

#TASK_NAME='ICSHM_DMG_UNET_srn101_small'
# create_unet_fn = lambda: sm.Unet("seresnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS,LEARNING_RATE=0.00005)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)

#TASK_NAME='ICSHM_STRUCT_UNET_inceptionv3_small'
# create_unet_fn = lambda: sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename,TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# # multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
#
#TASK_NAME='ICSHM_DMG_UNET_inceptionv3_small'
# create_unet_fn = lambda: sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)


#TASK_NAME='ICSHM_STRUCT_CUSTOM_UNET_small'
# create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=5, filters=24, num_classes=4, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)
#
#
#TASK_NAME='ICSHM_DMG_CUSTOM_UNET_small'
# create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=5, filters=24, num_classes=3, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename,TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)
#
#
#TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_np_small'
# create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=False)
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename,TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)


#TASK_NAME='ICSHM_DMG_DEEPLABV3p_np_small'
# create_model_fn = lambda: DeeplabV3Plus((RES_Y, RES_X, 3), 3, output_activation="softmax",is_pretrained=False)
# create_dmg_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_dmg_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=4)
#
#
#TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_small'
# create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=True)
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations  )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations, nrows=5)






# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn18_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_18_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn18_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_18_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )
#
# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn50_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_50_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )

# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn50_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_50_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )

# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn101_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_101_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )

# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn101_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_101_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )
#
# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn152_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )


# TRANSFER LEARNING


# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn101_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_101_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentations_all, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn101_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_101_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentations_all, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn152_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 4 )
# create_struct_task_fn = lambda model_basename, model, augmentations_all, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_small'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentations_all, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_rn101_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_rn101_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_rn152_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet152", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_rn152_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet152", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_inception3_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_inception3_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb4_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_efnb4_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb6_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCTsmall', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)
#
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_small'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='DMGsmall',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)


