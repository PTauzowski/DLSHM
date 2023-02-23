import os
import cv2 as cv
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter, ICSHM_DMG_Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from DLImages.convert import *
from DLModels.trainer import *
from DLGenerators.generators import *
import tensorflow as tf
from keras_unet.models import custom_unet

import matplotlib
matplotlib.use('Agg')
from matplotlib import image as mpimg, pyplot as plt


task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathDMG = os.path.join(task_path, 'TrainSets/DMG')
DMGmodel_name='ICSHM_DMGw_L4_E200'

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=200
BATCH_SIZE=64
resX=320
resY=160
nCHANNELS=3
nCLASSES=3
nLAYERS=4

imgDMG_conv  = ICSHM_DMG_Converter(resX,resY)
data_manager = ICSHMDataManager( images_source_path, csv_ind=6 )
data_manager.convertDataToNumpyFormat(imgDMG_conv, train_pathDMG )

model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, num_classes=nCLASSES, output_activation="softmax")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy",  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(3)])
model.summary()

class_weights=[8.0, 1.0, 4.0]
dataSource = DataSource( train_pathDMG, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
#trainer = DLTrainer( DMGmodel_name, model, task_path, dataSource, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES)
trainer = DLTrainer( DMGmodel_name, None, task_path, dataSource, DataGeneratorFromNumpyFilesWeighted, DataGeneratorFromNumpyFilesWeighted, DataGeneratorFromNumpyFilesWeighted, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES, class_weights)
#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

model=trainer.model

print("Evaluate on test data")
results = model.evaluate(trainer.testGen, batch_size=1)
print("test results:", results)

def testDMGPostprocess(filename, x, y, result):
    fig, axp = plt.subplots(4, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 3):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i]>0.5)
        axp[i, 3].imshow(result[0, :, :, i])
    plt.savefig(filename)
    plt.close(fig)

trainer.testModel(testDMGPostprocess)
