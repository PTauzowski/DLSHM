import os
import cv2 as cv
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from DLImages.convert import *
from DLModels.trainer import *
from DLGenerators.generators import *
import tensorflow as tf



task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathRGB = os.path.join(task_path, 'TrainSets/RGB')
model_path='/Users/piotrek/Computations/Ai/models'
RGBmodel_name='ICSHM_RGB_L4_E200'
RGBDmodel_name='ICSHM_RGBD_L4_E200'

EPOCHS=200
BATCH_SIZE=64
resX=320
resY=160
nCHANNELS=3
nCLASSES=8
nLAYERS=4

model = tf.keras.models.load_model(os.path.join(model_path, 'ICSHM_RGBsmall_cu_L4_F32_E200'))
model.save(os.path.join(model_path, 'ICSHM_RGBsmall_cu_L4_F32_E200.h5'))
exit(0)

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')
imgRGB_conv  = ICSHM_RGB_Converter(resX,resY)
data_manager = ICSHMDataManager(images_source_path )
data_manager.convertDataToNumpyFormat(imgRGB_conv, train_pathRGB )

#model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, num_classes=nCLASSES, output_activation="softmax")
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(8)])
#model.summary()

dataSource = DataSource( train_pathRGB, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
trainer = DLTrainer( model_name, None, task_path, dataSource, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES)
#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

model=trainer.model

print("Evaluate on test data")
results = model.evaluate(trainer.testGen, batch_size=1)
print("test results:", results)

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

trainer.testModel(testRGBPostprocess)
