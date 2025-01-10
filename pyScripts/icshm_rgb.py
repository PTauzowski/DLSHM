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

import sys

User='Mariusz'
#User='Piotr'

CURRENT_MODEL_NAME= 'ICSHM_RGB_DeepLabV3_E100'


if User=='Mariusz':
    TASK_PATH = "D:/Datasets/Tokaido_Dataset" # sys.argv[1]
    result_path = 'F:/Python/DL4SHM_results'
    TRAIN_IMAGES_PATH = TASK_PATH + '/' + 'DL4SHM_trainSet'
    MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
    IMAGES_SOURCE_PATH = 'D:/Datasets/Tokaido_Dataset'
    PREDICT_DIR = result_path + '/' + 'Predictions'
    TEST_PATH = result_path + '/' + 'Test'

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

imgRGB_conv  = ICSHM_RGB_Converter(RES_X, RES_Y)    # konwersja na pliki npy - jak sa, to juz tego nie robi
data_manager = ICSHMDataManager(IMAGES_SOURCE_PATH) # na razie nie wiadomo
data_manager.convert_data_to_numpy_format( imgRGB_conv, TRAIN_IMAGES_PATH )  # powinno sie nie uruchamiac, jak sa npy

# model = custom_unet(input_shape=(RES_Y,RES_X,N_CHANNELS), num_layers=N_LAYERS, filters=N_FILTERS, num_classes=N_CLASSES, output_activation="softmax")
# model = custom_vgg19(input_shape=(RES_Y,RES_X,3))
# model = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(resY,resX,nCHANNELS),  classes=nCLASSES, classifier_activation="softmax")
model = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
# model = u_net_compiled(input_size=(RES_Y,RES_X,N_CHANNELS), n_filters=N_FILTERS, n_classes=N_FILTERS)


# Kompilacja modelu i wyswitlenie informacji:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
model.summary()

# Przetwarzanie danych do trenowania i stworzenie obiektu trenera (może być niepotrzebny)
dataSource = DataSource( TRAIN_IMAGES_PATH, train_ratio=0.8, validation_ratio=0.15, sampleSize=-1, shuffle=True)
trainer = DLTrainer(CURRENT_MODEL_NAME, model, TASK_PATH)  # Tu wchodzi model, ale można dać "none" i będzie próbował model wydobyć z katalogu

# Generatory danych do trenowania (podstawia dane, jak w tablicy) i walidacji:
train_gen = DataGeneratorFromNumpyFiles(dataSource.get_train_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)
validation_gen = DataGeneratorFromNumpyFiles(dataSource.get_validation_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES)

# Rozpoczęcie treningu (w używania wytrenowanego modelu komentujemy funkcje poniżej)
trainer.train(train_gen, validation_gen, EPOCHS, BATCH_SIZE)
trainer.plot_training_history()
# trainer.plotTrainingHistory()

model=trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

# Poniższe funkcje są używane tylko w przypadku trenowania nowych modeli
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

# Testowanie na danych testowych (nie walidacyjnych)
trainer.testModel(testDMGsegmentation)
trainer.compute_measures(PREDICT_DIR, segmentationRGBInputFileReader(RES_X, RES_Y), predictDMGsegmentation)
