import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"
#import keras.backend as K

#from dlshm.dlimages import data_processing
from dlshm.dlimages.data_processing import ICSHM_RGB_Converter,ICSHM_RGB_4_Converter, ICSHMDataManager, ICSHM_RGB_FULL_Converter, compute_class_weights
from dlshm.dlmodels.loss_functions import weighted_categorical_crossentropy
from dlshm.dlmodels.c_unet import custom_unet

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dlshm.dlimages.convert import *
from dlshm.dlmodels.trainer import *
from dlshm.dlmodels.custom_models import *
from dlshm.dlgenerators.generators import *
from dlshm.dlimages.augmentations import augment_all, augment_brightness, augment_contrast, augment_noise, \
    augment_gamma, augment_flip, augment_rotation, augment_dl3_rgb, augment_unet_rgb, augment_dl3_rgb_np, augment_cutmix
from skimage.transform import resize
import pandas as pd

import matplotlib as mpl

from dlshm.dlimages.postprocess import write_prediction_segmentated, write_prediction_segmentated2, write_smooth_masks,write_smooth_masks_refined, test_dmg_segmentation

import sys

import tensorflow as tf
import segmentation_models as sm

# print(tf.__version__)  # TensorFlow version
# print(tf.keras.__path__)

# import datetime as dt

#User='Mariusz'
User='Piotr'

# 'a' - augmented with flip
# 'a2' - augmented without flip
# 'o' - no augmentation, without flip, no weights
# 'ob' - no augmentation, without flip, no weights, saved best of training process
# 'obw' - no augmentation, without flip, saved best of training process, weighted
# 'ab' - augmentation, with flip, no weights, saved best of training process
# 'es' - early stop and LR update
# 'np' - not pretrained
# 'x' - Xception in use
# 'b' Bigger UNet model with N_LAYERS and N_FILTERS

# CURRENT_MODEL_NAME= 'ICSHM_RGB_DEEPLABV3_100'
# CURRENT_MODEL_NAME = 'ICSHM_RGB_DEEPLABV3p_150a2'


# CURRENT_MODEL_NAME= 'ICSHM_RGB_UNET_100'
# CURRENT_MODEL_NAME= 'ICSHM_RGB_VGG19_100'

RES_X=640
RES_Y=320
N_CHANNELS=3
N_CLASSES=4
N_LAYERS=6
N_FILTERS=24
LEARNING_RATE = 0.001

# model_unet = custom_unet(input_shape=(RES_Y,RES_X,N_CHANNELS), num_layers=N_LAYERS, filters=N_FILTERS, num_classes=N_CLASSES, output_activation="softmax")
# model_vgg19a = build_vgg19_segmentation_model(input_shape=(RES_Y,RES_X,3), num_classes=N_CLASSES)
# model_vgg19b = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(RES_Y,RES_X,N_CHANNELS),  classes=N_CLASSES, classifier_activation="softmax")
# model_deeplab1 = DeepLabV3_1((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
# model_deeplab2 = DeepLabV3_2((RES_Y, RES_X, N_CHANNELS), N_CLASSES)N_CLASSES
#model_deeplabv3p = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
#model_deeplabv3p101 = DeeplabV3Plus101((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
# model = u_net_compiled(input_size=(RES_Y,RES_X,N_CHANNELS), n_filters=N_FILTERS, n_classes=N_FILTERS)



def rgb_model_function( model_name, model, augment_fn, batch_size, epochs):
    train_gen = None
    validation_gen = None
    test_gen=None
    try:
        CURRENT_MODEL_NAME = model_name

        #CURRENT_MODEL_NAME= 'ICSHM_RGB_DEEPLABV3_100'
        #CURRENT_MODEL_NAME= 'ICSHM_RGB_DEEPLABV3p_150a2'
        #CURRENT_MODEL_NAME= 'ICSHM_RGB_UNET_100'
        #CURRENT_MODEL_NAME= 'ICSHM_RGB_VGG19_100'

        if User=='Mariusz':
            TASK_PATH = "D:/Datasets/Tokaido_Dataset" # sys.argv[1]
            TRAIN_IMAGES_PATH = TASK_PATH + '/' + 'DL4SHM_trainSet'
            MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
            IMAGES_SOURCE_PATH = 'D:/Datasets/Tokaido_Dataset'
            PREDICT_DIR = 'F:/Python/DL4SHM_results/Predictions'
            TEST_PATH = 'F:/Python/DL4SHM_results' + '/' + 'Test'

        elif User=="Piotr":
            TASK_PATH = "/home/piotrek/Computations/Ai/ICSHM" # sys.argv[1]
            #TASK_PATH = "h:\\DL\\ICSHM"  # sys.argv[1]
            MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
            #IMAGES_SOURCE_PATH = '/Users/piotrek/DataSets/Tokaido_dataset_share'
            IMAGES_SOURCE_PATH = '/home/piotrek/Computations/Ai/Data/Tokaido_dataset_share'
            #IMAGES_SOURCE_PATH = '/Users/piotrek/Computations/Ai/Data/Tokaido_dataset_share'
            #IMAGES_SOURCE_PATH = 'h:\\DL\\ICSHM\\DataSets\\Tokaido_dataset_share'
            PREDICTIONS_PATH=os.path.join( MODEL_PATH, 'Predictions' )
            #TRAIN_IMAGES_PATH= TASK_PATH + '/' + 'TrainSets/RGB'
            TRAIN_IMAGES_PATH = '/home/piotrek/Computations/Ai/ICSHM/TrainSet4'
            TEST_PATH = MODEL_PATH + '/' + 'Test'


        # info_fil
        # e = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')


        class SegmentationRGBInputFileReader:   # Reading of the SOURCE images.
            def __init__(self, resX, resY ):
                self.x = np.zeros((resY, resX, 3), dtype=np.float32)
                self.resX=resX
                self.resY=resY

            def __call__(self, filename):
                image = cv.imread(filename)
                image_array = resize(image, (self.resY, self.resX), anti_aliasing=True)
                self.x[:, :, 0] = image_array[:, :, 0]
                self.x[:, :, 1] = image_array[:, :, 1]
                self.x[:, :, 2] = image_array[:, :, 2]
                return self.x

        def predictDMGsegmentation(x, y):  # wizualizacja masek z sieci
            colors = np.array([
                [0, 0, 0],  # background
                [1, 0, 0],  # mask 1 (red)
                [0, 1, 0],  # mask 2 (green)
                [0, 0, 1]  # mask 3 (blue)
            ], dtype=np.float32)

            nmasks = y.shape[2]
            masks = colors[np.argmax(y, axis=-1)]

            alpha = 0.6
            result= cv.addWeighted(masks, 1-alpha, x, alpha, 0)
            return result

        EPOCHS=epochs
        BATCH_SIZE=batch_size

        TRAIN_DATA_RATIO=0.8
        TEST_DATA_RATIO=0.15
        CROSS_VALIDATION_FOLDS=6

        #CLASS_NAMES =["Nonbridge", "Slab", "Beam", "Column", "Nonstructural", "Rail", "Sleeper", "Other" ]

        CLASS_NAMES =["Nonstructural", "Slab", "Beam", "Column" ]

        #dir_files_processing('/Users/piotrek/Computations/Ai/ICSHM/Predictions/Photos/Images', ImageResizer(RES_X,RES_Y,'/Users/piotrek/Computations/Ai/ICSHM/Predictions/Photos/PredictionPhotos'))
        imgRGB_conv  = ICSHM_RGB_4_Converter(RES_X, RES_Y)    # konwersja na pliki npy - jak sa, to juz tego nie robi
        data_manager = ICSHMDataManager(IMAGES_SOURCE_PATH) # na razie nie wiadomo
        data_manager.convert_data_to_numpy_format( imgRGB_conv, TRAIN_IMAGES_PATH )  # powinno sie nie uruchamiac, jak sa npy

        # weights = compute_class_weights(y)
        print("Class Weights:", data_manager.weights)
        print("Sum   Weights:", sum(data_manager.weights))

        # class weights reflects not only pixel ratio but also class importance
        #class_weights = np.array([0.05, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 0.05])

        class_weights = np.array([0.07, 0.33, 0.35, 0.25])

        # Normalize the weights so that their sum is 1
        normalized_class_weights = class_weights / np.sum(class_weights)

        loss_fn = weighted_categorical_crossentropy(normalized_class_weights)


        # Przetwarzanie danych do trenowania i stworzenie obiektu trenera (może być niepotrzebny)
        dataSource = DataSource( TRAIN_IMAGES_PATH, train_ratio=0.7, validation_ratio=0.1, sampleSize=-1)
        trainer = DLTrainer(CURRENT_MODEL_NAME, model, TASK_PATH)  # Tu wchodzi model, ale można dać "none" i będzie próbował model wydobyć z katalogu

        model=trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

        # Kompilacja modelu i wyswitlenie informacji:
        #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
        model.summary()

        # Generatory danych do trenowania (podstawia dane, jak w tablicy) i walidacji:
        train_gen = DataGeneratorFromNumpyFiles(dataSource.get_train_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES, augmentation_fn=augment_fn)
        validation_gen = DataGeneratorFromNumpyFiles(dataSource.get_validation_set_files(),1,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)
        test_gen = DataGeneratorFromNumpyFiles(dataSource.get_test_set_files(),1,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)

        #gener_test(os.path.join('/home/piotrek/Computations/Ai/ICSHM/Previews', CURRENT_MODEL_NAME), train_gen, scope=100)

        # Rozpoczęcie treningu (w używania wytrenowanego modelu komentujemy funkcje poniżej)
        trainer.train(train_gen, validation_gen, EPOCHS, BATCH_SIZE)
        trainer.plot_training_history()

        # Poniższe funkcje są używane tylko w przypadku trenowania nowych modeli
        print("Evaluate on test data")
        results = model.evaluate(test_gen, batch_size=1)
        print("test results:", results)


        # Testowanie na danych testowych (nie walidacyjnych)Unet-test
        trainer.test_model(test_gen,test_dmg_segmentation)
        dfs, df = trainer.compute_gen_measures(test_gen,class_weights,CLASS_NAMES)


        with pd.ExcelWriter(TASK_PATH+'/' + CURRENT_MODEL_NAME + '.xlsx', engine='openpyxl') as writer:
            dfs.to_excel(writer, sheet_name='ICSHM', index=False)
            df.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)

        #gener_test(os.path.join( '/Users/piotrek/Computations/Ai/ICSHM/Previews', CURRENT_MODEL_NAME), train_gen, scope=100)
        trainer.predict('/home/piotrek/Computations/Ai/ICSHM/Photos/PredictionPhotos',write_prediction_segmentated2)
        del train_gen, validation_gen, test_gen

    finally:
        # Cleanup
        import tensorflow.keras.backend as K
        import gc
        K.clear_session()
        gc.collect()

def multi_augmentation_training_structural(model_basename, create_model_fn, batch_size, epochs):
    tf.keras.backend.clear_session()
    print("* MULTI augmented training for model :",model_basename)
    model = create_model_fn()
    rgb_model_function( model_basename+'_none', model, None, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_br', model, augment_brightness, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_cn', model, augment_contrast, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_gm', model, augment_gamma, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_ns', model, augment_noise, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_fl', model, augment_flip, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function( model_basename+'_rot', model, augment_rotation, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = model = create_model_fn()
    rgb_model_function(model_basename + '_cut', model, augment_cutmix, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    model = create_model_fn()
    rgb_model_function(model_basename + '_all', model, augment_all, batch_size=batch_size, epochs=epochs)
    tf.keras.backend.clear_session()
    del model
    gc.collect()




import gc

# available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
#                   'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121',
#                   'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
#                   'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

epochs=200
batch_size=32

model_basename='ICSHM_RGB_UNET_rn18'
create_unet_sm = lambda: sm.Unet("resnet18", input_shape=(RES_Y, RES_X, N_CHANNELS), encoder_weights="imagenet", classes=4, activation="softmax")
#multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_unet_sm, batch_size=batch_size, epochs=epochs)

model_basename='ICSHM_RGB_UNET_rn101'
create_unet_sm = lambda: sm.Unet("resnet101", input_shape=(RES_Y, RES_X, N_CHANNELS), encoder_weights="imagenet", classes=4, activation="softmax")
multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_unet_sm, batch_size=batch_size, epochs=epochs)

model_basename='ICSHM_RGB_UNET_in3'
create_unet_sm = lambda: sm.Unet("inceptionv3", input_shape=(RES_Y, RES_X, N_CHANNELS), encoder_weights="imagenet", classes=4, activation="softmax")
multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_unet_sm, batch_size=batch_size, epochs=epochs)


# model_basename='ICSHM_RGB_UNET70'
# create_custom_unet = lambda : custom_unet(input_shape=(RES_Y,RES_X,N_CHANNELS), num_layers=N_LAYERS, filters=N_FILTERS, num_classes=N_CLASSES, output_activation="softmax")
# multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_custom_unet, batch_size=batch_size, epochs=epochs)
#
# model_basename='ICSHM_RGB_DEEPLABV3p'
# create_model_deeplab = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES, output_activation="softmax",is_pretrained=True)
# multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_model_deeplab, batch_size=batch_size, epochs=epochs)
#
# model_basename='ICSHM_RGB_DEEPLABV3p_np'
# create_model_deeplab = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES, output_activation="softmax",is_pretrained=False)
# multi_augmentation_training_structural(model_basename = model_basename, create_model_fn=create_model_deeplab, batch_size=batch_size, epochs=epochs)




