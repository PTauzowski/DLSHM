import os
import cv2 as cv
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from DLImages.convert import *
from DLModels.trainer import *
from DLGenerators.generators import *
import tensorflow as tf
from keras_unet.models import custom_unet


task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathRGB = os.path.join(task_path, 'TrainSets/RGB')
train_pathRGBD = os.path.join(task_path, 'TrainSets/RGBD')
train_pathDepth = os.path.join(task_path, 'TrainSets/Depth')
RGBmodel_name='ICSHM_RGB_L4_E200'
RGBDmodel_name='ICSHM_RGBD_L4_E200'
Depthmodel_name='ICSHM_Depth_L5_E200'

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=200
BATCH_SIZE=64
resX=320
resY=160
nCHANNELS=3
nCLASSES=8
nLAYERS=4

dataSourceRGB = DataSource( train_pathRGB, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
dataSourceRGBD = DataSource( train_pathRGBD, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
dataSourceDepth = DataSource( train_pathDepth, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
trainerRGB = DLTrainer( RGBmodel_name, None, task_path, dataSourceRGB, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), 3, nCLASSES)
trainerRGBD = DLTrainer( RGBDmodel_name, None, task_path, dataSourceRGBD, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), 4, nCLASSES)
trainerDepth = DLTrainer( Depthmodel_name, None, task_path, dataSourceDepth, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), 3, 1)
#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

print("Evaluate RGB on test data")
results = trainerRGB.model.evaluate(trainerRGB.testGen, batch_size=1)
print("test results:", results)

print("Evaluate RGBD on test data")
results = trainerRGBD.model.evaluate(trainerRGBD.testGen, batch_size=1)
print("test results:", results)

print("Evaluate Depth on test data")
results = trainerDepth.model.evaluate(trainerDepth.testGen, batch_size=1)
print("test results:", results)



class testRGBHybridPostprocess:

    def __init__(self,RGBmodel,RGBDmodel,Depthmodel):
        self.RGBmodel=RGBmodel
        self.RGBDmodel = RGBDmodel
        self.Depthmodel = Depthmodel

    def __call__(self, filename, x, y, resultRGBD ):
        fig, axp = plt.subplots(8, 5)
        fig.set_size_inches((20, 10))
        yRGB = self.RGBmodel.predict(x[:,:,:,0:3])
        yDepth = self.Depthmodel.predict(x[:,:,:,0:3])
        x[:,:,:,3:4]=yDepth
        pRGBD = self.RGBDmodel.predict(x)
        for i in range(0, 8):
            axp[i, 0].imshow(x[0, :, :, 0:3])
            axp[i, 1].imshow(y[0, :, :, i])
            axp[i, 2].imshow(yRGB[0, :, :, i]>0.5)
            axp[i, 3].imshow(pRGBD[0, :, :, i]>0.5)
            axp[i, 4].imshow(resultRGBD[0, :, :, i]>0.5)
        plt.savefig(filename)
        plt.close(fig)


trainerRGBD.testModel(testRGBHybridPostprocess(trainerRGB.model,trainerRGBD.model,trainerDepth.model))
