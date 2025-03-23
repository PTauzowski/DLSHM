import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from dlshm.dlimages import data_processing
from dlshm.dlimages.data_processing import ICSHM_DMG_Converter, ICSHMDataManager, ICSHM_RGB_FULL_Converter, compute_class_weights
from dlshm.dlmodels.loss_functions import weighted_categorical_crossentropy

import pandas as pd
from dlshm.dlimages.convert import *
from dlshm.dlmodels.trainer import *
from dlshm.dlmodels.custom_models import *
from dlshm.dlmodels.c_unet import custom_unet
from dlshm.dlgenerators.generators import *
import tensorflow as tf
from skimage.transform import resize
import pandas as pd
import matplotlib as mpl


# User='Mariusz'
User='Piotr'

# 'a' - augmented with flip
# 'a2' - augmented without flip
# 'o' - no augmentation, without flip, no weights
# 'ob' - no augmentation, without flip, no weights, saved best of training process
# 'obw' - no augmentation, without flip, saved best of training process, weighted
# 'ab' - augmentation, with flip, no weights, saved best of training process


CURRENT_MODEL_NAME= 'ICSHM_DMG_DEEPLABV3p_200obw'

if User=='Mariusz':
    TASK_PATH = "D:/Datasets/Tokaido_Dataset" # sys.argv[1]
    TRAIN_IMAGES_PATH = TASK_PATH + '/' + 'DL4SHM_trainSet'
    MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
    IMAGES_SOURCE_PATH = 'D:/Datasets/Tokaido_Dataset'
    PREDICT_DIR = 'F:/Python/DL4SHM_results/Predictions'
    TEST_PATH = 'F:/Python/DL4SHM_results' + '/' + 'Test'

elif User=="Piotr":
    #TASK_PATH = "/Users/piotrek/Computations/Ai/ICSHM" # sys.argv[1]
    TASK_PATH = "/home/piotrek/Computations/Ai/ICSHM"
    #TASK_PATH = "h:\\DL\\ICSHM"  # sys.argv[1]
    MODEL_PATH = TASK_PATH + '/' + CURRENT_MODEL_NAME
    #IMAGES_SOURCE_PATH = '/Users/piotrek/DataSets/Tokaido_dataset_share'
    IMAGES_SOURCE_PATH = '/Users/piotrek/Computations/Ai/Data/Tokaido_dataset_share'
    IMAGES_SOURCE_PATH = '/home/piotrek/Computations/Ai/Data/Tokaido_dataset_share'
    #IMAGES_SOURCE_PATH = 'h:\\DL\\ICSHM\\DataSets\\Tokaido_dataset_share'
    PREDICTIONS_PATH=os.path.join( MODEL_PATH, 'Predictions' )
    #TRAIN_IMAGES_PATH= TASK_PATH + '/' + 'TrainSets/RGB'
    TRAIN_IMAGES_PATH = '/home/piotrek/Computations/Ai/ICSHM/TrainSet/DMG'
    TEST_PATH = MODEL_PATH + '/' + 'Test'

#info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')

CLASS_NAMES = [ "background", "cracks", "reinforcement" ]

EPOCHS=200
BATCH_SIZE=16
RES_X=640
RES_Y=320
N_CHANNELS=3
N_CLASSES=3
N_LAYERS=5
N_FILTERS=21
LEARNING_RATE = 0.001

imgDMG_conv  = ICSHM_DMG_Converter(RES_X,RES_Y)
data_manager = ICSHMDataManager( IMAGES_SOURCE_PATH, csv_ind=6 )
data_manager.convert_data_to_numpy_format(imgDMG_conv, TRAIN_IMAGES_PATH)

model_unet = custom_unet(input_shape=(RES_Y,RES_X,N_CHANNELS), num_layers=N_LAYERS, filters=N_FILTERS, num_classes=N_CLASSES, output_activation="softmax")
model_vgg19a = build_vgg19_segmentation_model(input_shape=(RES_Y,RES_X,3), num_classes=N_CLASSES)
model_vgg19b = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(RES_Y,RES_X,N_CHANNELS),  classes=N_CLASSES, classifier_activation="softmax")
model_deeplab1 = DeepLabV3_1((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
model_deeplab2 = DeepLabV3_2((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
model_deeplabv3p = DeeplabV3Plus((RES_Y, RES_X, N_CHANNELS), N_CLASSES)
model_deeplabv3p101 = DeeplabV3Plus101((RES_Y, RES_X, N_CHANNELS), N_CLASSES)

model = model_deeplabv3p
model.summary()

# weights = compute_class_weights(y)
print("Class Weights:", data_manager.weights)
print("Sum   Weights:", sum(data_manager.weights))

class_weights= np.array([ 0.00174144, 0.09980335, 0.8984552 ])
normalized_class_weights = class_weights / np.sum(class_weights)

loss_fn = weighted_categorical_crossentropy(normalized_class_weights)

dataSource = DataSource( TRAIN_IMAGES_PATH, train_ratio=0.7, validation_ratio=0.1, sampleSize=-1, shuffle=True)
trainer = DLTrainer(CURRENT_MODEL_NAME, model, TASK_PATH)  # Tu wchodzi model, ale można dać "none" i będzie próbował model wydobyć z katalogu

model=trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

# Kompilacja modelu i wyswitlenie informacji:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
model.summary()

# Generatory danych do trenowania (podstawia dane, jak w tablicy) i walidacji:
train_gen = DataGeneratorFromNumpyFiles(dataSource.get_train_set_files(),BATCH_SIZE,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES, Augmentation=False)
validation_gen = DataGeneratorFromNumpyFiles(dataSource.get_validation_set_files(),1,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES, Augmentation=False)
test_gen = DataGeneratorFromNumpyFiles(dataSource.get_test_set_files(),1,(RES_Y,RES_X),(RES_Y,RES_X),N_CHANNELS,N_CLASSES, Augmentation=False)

# Rozpoczęcie treningu (w używania wytrenowanego modelu komentujemy funkcje poniżej)
trainer.train(train_gen, validation_gen, EPOCHS, BATCH_SIZE)
trainer.plot_training_history()


print("Evaluate on test data")
results = model.evaluate(test_gen, batch_size=1)
print("test results:", results)

def test_dmg_postprocess(filename, x, y, result):
    fig, axp = plt.subplots(3, 4)
    fig.set_size_inches((20, 10))
    for i in range(0, 3):
        axp[i, 0].imshow(x[0, :, :, :])
        axp[i, 1].imshow(y[0, :, :, i])
        axp[i, 2].imshow(result[0, :, :, i]>0.5)
        axp[i, 3].imshow(result[0, :, :, i]>0.1)
    plt.savefig(filename)
    plt.close(fig)

def test_dmg_segmentation(pathname, x, y, result):
    path, filename = os.path.split(pathname)
    name, extension = os.path.splitext(filename)
    source_name=os.path.join(path,name+"_source")+extension
    test_name = os.path.join(path,name + "_result")+extension
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

    accuracy =  np.mean( y == (result > 0.5).astype(int) )
    epsilon = 1.0E-07
    y_pred = tf.clip_by_value(result, clip_value_min=epsilon, clip_value_max=1.0 - epsilon)
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=-1))

    nmasks = y.shape[3]
    masks = colors[np.argmax(result, axis=-1)]
    sourse_masks = colors[np.argmax(y, axis=-1)]

    alpha = 0.6
    blended = cv.addWeighted(masks[0,], 1-alpha, x[0,], alpha, 0)
    source_blended = cv.addWeighted(sourse_masks[0,], 1 - alpha, x[0,], alpha, 0)
    # Display the result in a window
    cv.imwrite(source_name, source_blended*255)
    cv.imwrite(test_name,blended*255)
    #print(name," accuracy = ",accuracy," loss = ", loss, "\n")


 # Testowanie na danych testowych (nie walidacyjnych)
trainer.test_model(test_gen,test_dmg_segmentation)
dfs, df = trainer.compute_gen_measures(test_gen,class_weights,CLASS_NAMES)

with pd.ExcelWriter(TASK_PATH+'/' + CURRENT_MODEL_NAME + '.xlsx', engine='openpyxl') as writer:
   dfs.to_excel(writer, sheet_name='ICSHM', index=False)
   df.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)
