import os
import cv2 as cv
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from dlimages.convert import *
from dlmodels.trainer import *
from dlgenerators.generators import *
import tensorflow as tf
from keras_unet.models import custom_unet
from matplotlib import image as mpimg, pyplot as plt



task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathRGBD = os.path.join(task_path, 'TrainSets/RGBD')
model_name='ICSHM_RGBD_L4_E200'

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=200
BATCH_SIZE=32
resX=320
resY=160
nCHANNELS=4
nCLASSES=8
nLAYERS=4

imgRGB_conv  = ICSHM_RGB_Converter(resX,resY)
imgRGBD_conv  = ICSHM_RGBD_Converter(resX,resY)
data_manager = ICSHMDataManager(images_source_path )
data_manager.convert_data_to_numpy_format(imgRGBD_conv, train_pathRGBD)

model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, num_classes=nCLASSES, output_activation="softmax")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy",  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(8)])
model.summary()

dataSource = DataSource(train_pathRGBD, train_ratio=0.8, validation_ratio=0.15, sampleSize=-1, shuffle=True)
trainer = DLTrainer( model_name, None, task_path, dataSource, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES)

model=trainer.model

#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

def testRGBPostprocess(filename, x, y, result):
    fig, axp = plt.subplots(8, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 8):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i]>0.5)
        axp[i, 3].imshow(result[0, :, :, i])
    plt.savefig(filename)
    del fig

trainer.test_model(testRGBPostprocess)

