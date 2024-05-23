import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import image as mpimg, pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2 as cv

from dlmodels.keras_depth_model import DepthEstimationModel
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter, ICSHM_Depth_Converter

import pandas as pd
from dlimages.convert import *
from dlmodels.trainer import *
from dlgenerators.generators import *
import tensorflow as tf
from keras_unet.models import custom_unet



task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathDepth = os.path.join(task_path, 'TrainSets/Depth')
model_name='ICSHM_Depth_L5_E200'

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=200
BATCH_SIZE=64
resX=320
resY=160
nCHANNELS=3
nCLASSES=1
nLAYERS=5
LR = 0.001


imgDepth_converter  = ICSHM_Depth_Converter(resX,resY)
data_manager = ICSHMDataManager(images_source_path )
data_manager.convert_data_to_numpy_format(imgDepth_converter, train_pathDepth)

model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, num_classes=nCLASSES, output_activation="sigmoid")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mean_absolute_error",  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanIoU(2)])
model.summary()

dataSource = DataSource( train_pathDepth, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
trainer = DLTrainer( model_name, None, task_path, dataSource, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES)

#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

model=trainer.model

print("Evaluate on test data")
results = model.evaluate(trainer.testGen, batch_size=1)
print("test results:", results)

def testDepthPostprocess(filename, x, y, result):
    fig, axp = plt.subplots(1, 3)
    fig.set_size_inches((20, 10))
    axp[0].imshow(x[0, :, :])
    axp[1].imshow(y[0, :, :])
    axp[2].imshow(result[0, :, :])
    plt.savefig(filename)
    del fig

trainer.test_model(testDepthPostprocess)

