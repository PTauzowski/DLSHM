import os
import cv2 as cv
from ICSHM import DataProcessing
from ICSHM.DataProcessing import ICSHM_RGB_Converter, ICSHMDataManager, ICSHM_RGBD_Converter, ICSHM_DMGC_Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from dlimages.convert import *
from dlmodels.trainer import *
from dlmodels.custom_models import *
from dlgenerators.generators import *
import tensorflow as tf
from keras_unet.models import custom_unet

import matplotlib
matplotlib.use('Agg')
from matplotlib import image as mpimg, pyplot as plt


task_path = 'H:/DL/ICSHM'
images_source_path = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share'
data_info_file = 'H:/DL/ICSHM/DataSets/Tokaido_dataset_share/files_train.csv'
train_pathDMGC = os.path.join(task_path, 'TrainSets/DMGCbig')
DMGmodel_name='ICSHM_DMGCbig_DeepLabV3at_E100'

info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

EPOCHS=100
BATCH_SIZE=4
resX=640
resY=320
nCHANNELS=3
nCLASSES=2
nFILTERS=32
nLAYERS=4

imgDMGC_conv  = ICSHM_DMGC_Converter(resX,resY)
data_manager = ICSHMDataManager( images_source_path, csv_ind=6 )
data_manager.convert_data_to_numpy_format(imgDMGC_conv, train_pathDMGC)

#model = custom_unet(input_shape=(resY,resX,nCHANNELS), num_layers=nLAYERS, filters=nFILTERS, num_classes=nCLASSES, output_activation="softmax")
model = DeeplabV3Plus((resY,resX,nCHANNELS), nCLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(2)])
model.summary()

class_weights=[1.0, 0.1]
dataSource = DataSource( train_pathDMGC, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True )
#trainer = DLTrainer( DMGmodel_name, model, task_path, dataSource, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, DataGeneratorFromNumpyFiles, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES)
trainer = DLTrainer( DMGmodel_name, None, task_path, dataSource, DataGeneratorFromNumpyFilesWeighted, DataGeneratorFromNumpyFilesWeighted, DataGeneratorFromNumpyFilesWeighted, BATCH_SIZE, (resY,resX), nCHANNELS, nCLASSES, class_weights)
#trainer.train(EPOCHS,BATCH_SIZE)
#trainer.plotTrainingHistory()

model=trainer.model
#trainer.saveModel()

print("Evaluate on test data")
results = model.evaluate(trainer.testGen, batch_size=1)
print("test results:", results)

def testDMGPostprocess(filename, x, y, result):
    fig, axp = plt.subplots(4, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 2):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i]>0.5)
        axp[i, 3].imshow(result[0, :, :, i])
    plt.savefig(filename)
    plt.close(fig)

def testDMGsegmentation(pathname, x, y, result):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    # Define the color palette for the segmentation masks
    colors = np.array([
        [0, 0, 1],  # background
        [0, 0, 0],  # mask 1 (red)
        [0, 1, 0],  # mask 2 (green)
        [0, 0, 1],  # mask 3 (blue)
        [1, 1, 0],  # mask 4 (yellow)
        [1, 0, 1],  # mask 5 (magenta)
        [0, 1, 1],  # mask 6 (cyan)
        [0.5, 0.5, 0.5],  # mask 7 (gray)
    ], dtype=np.float32)

    nmasks = y.shape[3]
    masks = colors[np.argmax(result, axis=-1)]

    alpha = 0.6
    blended = cv.addWeighted(masks[0,], 1-alpha, x[0,], alpha, 0)
    # Display the result in a window
    cv.imwrite(pathname,blended*255)

trainer.test_model_weighted(testDMGsegmentation)
