import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
import keras.backend as K

from dlimages import data_processing
from dlimages.data_processing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGB_FULL_Converter
from dlmodels.unet import u_net_compiled

import pandas as pd
from dlimages.convert import *
from dlmodels.trainer import *
from dlmodels.custom_models import *
from dlgenerators.generators import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from skimage.transform import resize
from keras_unet.models import custom_unet


TASK_PATH = '/Users/piotrek/Computations/Ai/ICSHM'
IMAGES_SOURCE_PATH = '/Users/piotrek/DataSets/Tokaido_dataset_share'
DATA_INFO_FILE = '/Users/piotrek/DataSets/Tokaido_dataset_share/files_train.csv'
TRAIN_PATH_RGB = os.path.join(TASK_PATH, 'TrainSet')
PREDICT_DIR=os.path.join(TASK_PATH, 'Predictions')
RGB_MODEL_NAME= 'ICSHM_RGB_DeepLabV3_E100'

# info_fil
# e = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')



class segmentationRGBInputFileReader:
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

def predictDMGsegmentation(x, y):
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

EPOCHS=100
BATCH_SIZE=16
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

imgRGB_conv  = ICSHM_RGB_Converter(RES_X, RES_Y)
data_manager = ICSHMDataManager(IMAGES_SOURCE_PATH)
data_manager.convert_data_to_numpy_format( imgRGB_conv, TRAIN_PATH_RGB )

#model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, filters=nFILTERS, num_classes=nCLASSES, output_activation="softmax")
#model = custom_vgg19(input_shape=(resY,resX,3))
#model = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(resY,resX,nCHANNELS),  classes=nCLASSES, classifier_activation="softmax")
model = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
#model=UNetCompiled(input_size=(resY,resX,nCHANNELS), n_filters=nFILTERS, n_classes=nCLASSES)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
model.summary()

dataSource = DataSource( TRAIN_PATH_RGB, train_ratio=0.8, validation_ratio=0.15, sampleSize=-1, shuffle=True)
trainer = DLTrainer(RGB_MODEL_NAME, model, TASK_PATH)

train_gen = DataGeneratorFromNumpyFiles(dataSource.get_train_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)
validation_gen = DataGeneratorFromNumpyFiles(dataSource.get_validation_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)

trainer.train(train_gen, validation_gen, EPOCHS, BATCH_SIZE)
trainer.plotTrainingHistory()

model=trainer.model

# print("Evaluate on test data")
# results = model.evaluate(trainer.testGen, batch_size=1)
# print("test results:", results)

def testDMGsegmentation(pathname, x, y, result):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    img_name=os.path.join(path,name+"_image")+extension
    # Define the color palette for the segmentation masks
    colors = np.array([
        [0, 0, 0],  # background
        [1, 0, 0],  # mask 1 (red)
        [0, 1, 0],  # mask 2 (green)
        [0, 0, 1],  # mask 3 (blue)
        [1, 1, 0],  # mask 4 (yellow)
        [1, 0, 1],  # mask 5 (magenta)
        [0, 1, 1],  # mask 6 (cyan)
        [0.5, 0.5, 0.5],  # mask 7 (gray)
    ], dtype=np.float32)

    accuracy =  np.mean( y == (result > 0.5).astype(np.int) )
    epsilon = K.epsilon()
    y_pred = K.clip(result, epsilon, 1.0 - epsilon)
    loss = K.mean(-K.sum(y * K.log(y_pred), axis=-1))

    nmasks = y.shape[3]
    masks = colors[np.argmax(result, axis=-1)]
    sourse_masks = colors[np.argmax(y, axis=-1)]

    alpha = 0.6
    blended = cv.addWeighted(masks[0,], 1-alpha, x[0,], alpha, 0)
    source_blended = cv.addWeighted(sourse_masks[0,], 1 - alpha, x[0,], alpha, 0)
    # Display the result in a window
    cv.imwrite(pathname,blended*255)
    cv.imwrite(img_name, source_blended*255)
    print(name," accuracy = ",accuracy," loss = ", loss, "\n")

def testRGBPostprocess(filename, x, y, result):
    fig, axp = plt.subplots(8, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 8):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i]>0.5)
        axp[i, 3].imshow(result[0, :, :, i])
    plt.savefig(filename)
    plt.close(fig)

trainer.testModel(testDMGsegmentation)
trainer.compute_measures(PREDICT_DIR, segmentationRGBInputFileReader(RES_X, RES_Y), predictDMGsegmentation)
