import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
#import keras.backend as K

#from dlshm.dlimages import data_processing
from dlshm.dlimages.data_processing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGB_FULL_Converter
from dlshm.dlmodels.unet import u_net_compiled

import pandas as pd
from dlshm.dlimages.convert import *
from dlshm.dlmodels.trainer import *
from dlshm.dlmodels.custom_models import *
from dlshm.dlgenerators.generators import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from skimage.transform import resize
from keras_unet.models import custom_unet

import datetime

import sys

t0 = datetime.datetime.now()

User='Mariusz'
#User='Piotr'

CURRENT_MODEL_NAME= 'ICSHM_RGB_DeepLabV3_E100'


if User=='Mariusz':
    TASK_PATH = "D:/Datasets/Tokaido_Dataset" # sys.argv[1]
    TRAIN_IMAGES_PATH = TASK_PATH + '/' + 'DL4SHM_trainSet'
    MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
    IMAGES_SOURCE_PATH = 'D:/Datasets/Tokaido_Dataset'
    PREDICT_DIR = 'F:/Python/DL4SHM_results/Predictions'
    TEST_PATH = 'F:/Python/DL4SHM_results' + '/' + 'Test'

elif User=="Piotr":
    TASK_PATH = "/Users/piotrek/Computations/Ai/ICSHM" # sys.argv[1]
    MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
    IMAGES_SOURCE_PATH = '/Users/piotrek/DataSets/Tokaido_dataset_share'
    PREDICTIONS_PATH=os.path.join( MODEL_PATH, 'Predictions' )
    TRAIN_IMAGES_PATH= TASK_PATH + '/' + 'TrainSet'
    TEST_PATH = MODEL_PATH + '/' + 'Test'


# info_fil
# e = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')


class segmentationRGBInputFileReader:   # Reading of the SOURCE images.
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
        [0, 0, 1],  # mask 3 (blue)
        [1, 1, 0],  # mask 4 (yellow)
        [1, 0, 1],  # mask 5 (magenta)
        [0, 1, 1],  # mask 6 (cyan)
        [0, 0, 0],  # mask 7 (gray)
    ], dtype=np.float32)

    nmasks = y.shape[2]
    masks = colors[np.argmax(y, axis=-1)]

    alpha = 0.6
    result= cv.addWeighted(masks, 1-alpha, x, alpha, 0)
    return result

EPOCHS=3
BATCH_SIZE=50
RES_X=640
RES_Y=320
N_CHANNELS=3
N_CLASSES=8
N_LAYERS=5
N_FILTERS=21
LEARNING_RATE = 0.001

TRAIN_DATA_RATIO=0.8
TEST_DATA_RATIO=0.15
CROSS_VALIDATION_FOLDS=6

model = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES)

t1 = datetime.datetime.now()

print("model:", t1 - t0)

# Kompilacja modelu i wyswitlenie informacji:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss="categorical_crossentropy", 
              metrics=[tf.keras.metrics.CategoricalAccuracy(), 
                       tf.keras.metrics.MeanIoU(N_CLASSES)])

t2 = datetime.datetime.now()
print("Kompilacja modelu:", t2 - t1)

model.summary()
