import glob
import os

import keras_hub

from dlshm.dlimages.augmentations import augment_brightness, augment_flip, augment_contrast, augment_gamma, \
    augment_noise, augment_all, augment_cutmix
from dlshm.dlmodels.basnet import BASNet
from dlshm.dlmodels.custom_models import DeeplabV3Plus, create_deeplab_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"


import tensorflow as tf

import tensorflow as tf
print(tf.__version__)
print(tf.keras)

from tensorflow.keras import layers, models

from dlshm.dlimages.data_processing import ICSHM_DMG_Converter, ICSHMDataManager, ICSHM_STRUCTD_Converter, \
    ICSHM_STRUCT_Converter
from dlshm.dlmodels.c_unet import custom_unet
from dlshm.dlresults.postprocess import prepare_excel_multiaugmented_results


import tensorflow as tf
import segmentation_models as sm
from dlshm.dlimages.ICSHM_tasks import ICSHM_structural_task, ICSHM_damage_task, multi_augmentation_training_structural, \
    multi_augmentation_transfer_learning, predict_photos_in_all_tasks, compute_measures

# available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
#                   'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121',
#                   'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
#                   'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

RES_X=640
RES_Y=320
BATCH_SIZE=16
TASK_PATH = '/home/piotrek/Computations/Ai/ICSHM'
SOURCE_PATH = '/home/piotrek/Computations/Ai/Data/Tokaido_dataset_share'
#PHOTO_TEST_PATH = '/home/piotrek/Computations/Ai/ICSHM/TestSet/PhotoTestSet'
PHOTO_TEST_PATH = '/home/piotrek/Computations/Ai/ICSHM/TestSet/Photos/Images'
PHOTO_NUMPY_TEST_PATH = '/home/piotrek/Computations/Ai/ICSHM/TestSet/PhotoTestSet'

# data_manager = ICSHMDataManager(SOURCE_PATH)
# data_manager.convert_folders_data_to_numpy_format( ICSHM_STRUCT_Converter(RES_X,RES_Y),
#                                                    '/Users/piotrek/Computations/Ai/ICSHM/TestSet/Photos/Images',
#                                                    '/Users/piotrek/Computations/Ai/ICSHM/TestSet/Photos/Masks',
#                                                    PHOTO_TEST_PATH)

data_manager = ICSHMDataManager(SOURCE_PATH)
data_manager.convert_folders_data_to_numpy_format( ICSHM_STRUCT_Converter(320,160),
                                                   '/home/piotrek/Computations/Ai/ICSHM/TestSet/Photos/Images',
                                                   '/home/piotrek/Computations/Ai/ICSHM/TestSet/Photos/Masks',
                                                   PHOTO_NUMPY_TEST_PATH)



augmentations =  (  ("none", "_none", None),
                    ("brightness", "_br", augment_brightness),
                    ("contrast", "_cn", augment_contrast),
                    ("gamma", "_gm", augment_gamma),
                    ("noise", "_ns", augment_noise),
                    ("flip", "_fl", augment_flip),
                    ("rotation", "_rot", augment_flip),
                    ("cutmix", "_cut", augment_cutmix),
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

# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_DEEPLABV3p_rn50_small', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_DEEPLABV3p_rn50_small', augmentations, nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_DEEPLABV3p_rn18_small', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_DEEPLABV3p_rn18_small', augmentations, nrows=4)
#
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_srn101_small', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_srn101_small', augmentations, nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_rn152_small', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_rn152_small', augmentations, nrows=4)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_STRUCT_UNET_efnb4_small', augmentations, nrows=5)
# prepare_excel_multiaugmented_results(TASK_PATH, 'ICSHM_DMG_UNET_efnb4_small', augmentations, nrows=4)


import keras
keras.config.disable_traceback_filtering()

#TASK_NAME = 'ICSHM_STRUCT_UNET_rn152_small_all'
#predict_photos_in_all_tasks(TASK_PATH,TASK_NAME,PHOTO_TEST_PATH,320,160)
#compute_measures(TASK_PATH, TASK_NAME, PHOTO_NUMPY_TEST_PATH, 320, 160,[1, 1, 1, 1],[ "Nonstructural", "Slab", "Beam", "Column" ])



# TASK_NAME='ICSHM_DMG_UNET_efnb5_1'
# #model = sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task = ICSHM_damage_task(model=None, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BATCH_SIZE, augmentation_fn=augment_all, LEARNING_RATE=0.00005)
# create_struct_task.compute_all_sets_measures()
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4'
# #model = sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task = ICSHM_damage_task(model=None, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BATCH_SIZE, augmentation_fn=augment_all, LEARNING_RATE=0.00005)
# create_struct_task.compute_all_sets_measures()

# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_all_metrics'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_1'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_2'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_3'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_4'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_5'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_6'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_7'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_8'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_all_metrics_9'
create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#



#
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky'
# model = reate_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task = ICSHM_damage_task(model=None, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BATCH_SIZE, augmentation_fn=augment_all, LEARNING_RATE=0.00005)
# create_struct_task.compute_all_sets_measures()
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152'
# model = reate_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task = ICSHM_damage_task(model=None, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BATCH_SIZE, augmentation_fn=augment_all, LEARNING_RATE=0.00005)
# create_struct_task.compute_all_sets_measures()



# TASK_NAME='ICSHM_STRUCT_BASNet_LR45cos2_all'
# #model = sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# model = BASNet( input_shape=(RES_Y, RES_X, 3), out_classes=4 )  # Create mod
# create_struct_task = ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BATCH_SIZE, augmentation_fn=augment_all, LEARNING_RATE=0.00005)
# create_struct_task.train()



# TASK_NAME='ICSHM_STRUCT_UNET_rn18'

# create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations  )


# TASK_NAME='ICSHM_STRUCT_UNET_rn18'
# create_unet_fn = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# #create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=24, num_classes=4, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmen##tation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
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

# TASK_NAME='ICSHM_STRUCT_CUSTOM_UNET'
# create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=16, num_classes=4, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_CUSTOM_UNET'
# create_unet_fn = lambda: custom_unet(input_shape=(RES_Y,RES_X,3), num_layers=6, filters=16, num_classes=3, output_activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_np'
# create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=False)
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_np'
# create_model_fn = lambda: DeeplabV3Plus((RES_Y, RES_X, 3), 3, output_activation="softmax",is_pretrained=False)
# create_dmg_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_dmg_task_fn, BATCH_SIZE  )

#
#
# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p'
# create_model_fn = lambda:  DeeplabV3Plus((RES_Y, RES_X, 3), 4, output_activation="softmax",is_pretrained=True)
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE  )
#
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p'
# create_model_fn = lambda: DeeplabV3Plus((RES_Y, RES_X, 3), 3, output_activation="softmax",is_pretrained=True)
# create_dmg_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, augmentation_fn=augmentation_fn)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_dmg_task_fn, BATCH_SIZE  )


# Load a trained backbone to extract features from it's `pyramid_outputs`.
# image_encoder = keras_hub.models.ResNetBackbone.from_preset(
#     "resnet_101_imagenet"
# )

# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p'
# create_model_fn = lambda:  keras_hub.models.DeepLabV3Backbone( image_encoder=image_encoder, projection_filters=48, low_level_feature_key="P2", spatial_pyramid_pooling_key="P5", upsampling_size = 8, dilation_rates = [6, 12, 18] )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations )

# TASK_NAME='ICSHM_STRUCT_DEEPLABV3p_rn152_1'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Struct', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )


# TASK_NAME='ICSHM_STRUCT_UNET_efnb4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb4", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS, LEARNING_RATE : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='STRUCT', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=LEARNING_RATE)
# multi_augmentation_transfer_learning(TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all  )
#prepare_excel_multiaugmented_results( TASK_PATH, TASK_NAME, augmentations_all, nrows=5)

# TASK_NAME='ICSHM_DMG_UNET_rn152'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet152", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_rn101'
# create_unet_fn = lambda: sm.Unet(backbone_name="resnet101", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn101'
# create_model_fn = lambda:  create_deeplab_model( "resnet_101_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_3'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_4'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_5'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )

# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_1'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_2'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_3'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_4'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_5'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_6'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_7'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_DEEPLABV3p_rn152_weighted_focal_tversky_8'
# create_model_fn = lambda:  create_deeplab_model( "resnet_152_imagenet", 3 )
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg', RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural(TASK_NAME, create_model_fn, create_struct_task_fn, BATCH_SIZE , augmentations=augmentations_all )


# TASK_NAME='ICSHM_DMG_UNET_efnb6_05'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.000005)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_1'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_2'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00002)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_STRUCT_UNET_efnb6_4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Struct',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb7_4p'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb7", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)

# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_1'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_2'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_3'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_5'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_6'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_7'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_8'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_9'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4_weighted_focal_tversky4_10'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )

# TASK_NAME='ICSHM_STRUCT_UNET_efnb7_4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb7", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=4, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_structural_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Struct',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)

# TASK_NAME='ICSHM_DMG_UNET_efnb6_6'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00006)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_8'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00008)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb6_10'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb6", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_05'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.000005)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_1'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_2'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00002)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_4'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00004)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_6'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00006)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_8'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.00008)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
# #prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)
#
# TASK_NAME='ICSHM_DMG_UNET_efnb5_10'
# create_unet_fn = lambda: sm.Unet(backbone_name="efficientnetb5", input_shape=(RES_Y, RES_X, 3), encoder_weights="imagenet", classes=3, activation="softmax")
# create_struct_task_fn = lambda model_basename, model, augmentation_fn, BS : ICSHM_damage_task(model=model, TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=model_basename, TRAIN_DIR='Dmg',RES_X=RES_X, RES_Y=RES_Y, BATCH_SIZE=BS, LEARNING_RATE=0.0001)
# multi_augmentation_training_structural( TASK_NAME, create_unet_fn, create_struct_task_fn, BATCH_SIZE, augmentations=augmentations_all )
#prepare_excel_multiaugmented_results(TASK_PATH, TASK_NAME, augmentations_all, nrows=4)