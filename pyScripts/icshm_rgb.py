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


# task_path = 'H:/DL/ICSHM'
task_path = '/Users/piotrek/Data/DocumentsUnix/Prezentacje/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathRGB = os.path.join(task_path, 'TrainSets/RGBbig')
predict_dir=os.path.join(task_path,'TestSets/V4photosWithMasks')
RGBmodel_name='ICSHM_RGB_DeepLabV3_E100'
# RGBmodel_name='ICSHM_RGB_UNET_13_E100'
#RGBmodel_name='ICSHM_RGB_VGG19_E50'

# info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=100
BATCH_SIZE=32
resX=640
resY=320
nCHANNELS=3
nCLASSES=8
nLAYERS=5
nFILTERS=21
LR = 0.001

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

imgRGB_conv  = ICSHM_RGB_Converter(resX,resY)
data_manager = ICSHMDataManager(images_source_path )
data_manager.convertDataToNumpyFormat(imgRGB_conv, train_pathRGB )

#model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, filters=nFILTERS, num_classes=nCLASSES, output_activation="softmax")
#model = custom_vgg19(input_shape=(resY,resX,3))
#model = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(resY,resX,nCHANNELS),  classes=nCLASSES, classifier_activation="softmax")
model = DeeplabV3Plus((resY,resX,nCHANNELS), nCLASSES)
#model=UNetCompiled(input_size=(resY,resX,nCHANNELS), n_filters=nFILTERS, n_classes=nCLASSES)

# change the last layer to output a tensor of shape (height, width, num_masks)
#outputs = Conv2D(nCLASSES, (resY, resX), activation='softmax')(x)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="categorical_crossentropy",  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(nCLASSES)])
model.summary()

# dataSource = DataSource( train_pathRGB, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
trainer = DLTrainer( RGBmodel_name, None, task_path, None, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, idim=(resY,resX), odim=(resY,resX), n_channels=nCHANNELS, n_classes=nCLASSES)

# trainer.train(EPOCHS,BATCH_SIZE)
# trainer.plotTrainingHistory()

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

#trainer.testModel(testDMGsegmentation)
trainer.compute_measures(predict_dir, segmentationRGBInputFileReader(resX, resY), predictDMGsegmentation)
