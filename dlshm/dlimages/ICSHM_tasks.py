import os
import gc
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 as cv
from keras.src.losses import CategoricalFocalCrossentropy

from dlshm.dlgenerators.generators import DataSource, DataGeneratorFromNumpyFiles
from dlshm.dlimages.augmentations import augment_brightness, augment_contrast, augment_gamma, augment_noise, \
    augment_rotation, augment_cutmix, augment_all, augment_flip
from dlshm.dlimages.postprocess import test_dmg_segmentation, write_prediction_segmentated2, \
    write_prediction_segmentated3
from dlshm.dlmodels import trainer
from dlshm.dlmodels.loss_functions import weighted_categorical_crossentropy, dice_loss, tversky_loss, \
    wrapped_tversky_loss, weighted_tversky_loss, focal_tversky_loss, weighted_focal_tversky_loss
from dlshm.dlimages.data_processing import ICSHM_STRUCT_Converter, ICSHM_DMG_Converter, ICSHMDataManager, \
    ICSHM_STRUCTD_Converter
from dlshm.dlmodels.trainer import DLTrainer
from pyScripts.icshm_rgb_batch import LEARNING_RATE


class ICSHM_Task:
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, N_CHANNELS=3, N_CLASSES=4, N_LAYERS=6, N_FILTERS=24, BATCH_SIZE=32, EPOCHS=200, LEARNING_RATE = 0.0001, augmentation_fn=None):
        self.model=model[0]
        self.RES_X = RES_X
        self.RES_Y = RES_Y
        self.N_CHANNELS = N_CHANNELS
        self.N_CLASSES = N_CLASSES
        self.N_LAYERS = N_LAYERS
        self.N_FILTERS = N_FILTERS
        self.BATCH_SIZE=BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.TASK_PATH = TASK_PATH
        self.SOURCE_PATH = SOURCE_PATH
        self.TASK_NAME = TASK_NAME
        self.EPOCHS = EPOCHS
        self.augmentation_fn = augmentation_fn

    def load_model(self,filename):
        self.model = tf.keras.models.load_model(filename,compile=False)
        print('Model ',self.TASK_PATH,' was found and loaded')

    def create_dataset(self,train_dir,converter):
        self.data_manager = ICSHMDataManager(self.SOURCE_PATH,csv_ind=self.csv_ind)
        self.TRAIN_PATH = os.path.join(self.TASK_PATH, train_dir)
        self.data_manager.convert_tokaido_data_to_numpy_format(converter, self.TRAIN_PATH)

    def train(self):
        self.dataSource = DataSource(self.TRAIN_PATH, train_ratio=0.70, validation_ratio=0.15 )
        self.trainer = DLTrainer(self.TASK_PATH, self.TASK_NAME, self.model)
        if not self.trainer.model_dir_exists:
            train_set, validation_set = self.dataSource.get_training_data()
            train_gen = DataGeneratorFromNumpyFiles(train_set, self.BATCH_SIZE, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES, augmentation_fn=self.augmentation_fn)
            validation_gen = DataGeneratorFromNumpyFiles(validation_set, 1, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES,shuffle=False)
            test_gen = DataGeneratorFromNumpyFiles(self.dataSource.get_test_files(), 1, (self.RES_Y, self.RES_X), (self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES,shuffle=False)
            model = self.trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

            # Kompilacja modelu i wyswitlenie informacji:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=self.loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(self.N_CLASSES)])
            # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss="categorical_crossentropy",
            #               metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(self.N_CLASSES)])
            model.summary()
            # gener_test(os.path.join('/home/piotrek/Computations/Ai/ICSHM/Previews', CURRENT_MODEL_NAME), train_gen, scope=100)

            # Rozpoczęcie treningu (w używania wytrenowanego modelu komentujemy funkcje poniżej)
            self.trainer.train(train_gen, validation_gen, self.EPOCHS, self.BATCH_SIZE)

            # Poniższe funkcje są używane tylko w przypadku trenowania nowych modeli
            print("Evaluate on test data")
            results = model.evaluate(test_gen, batch_size=1)
            print("test results:", results)

            # Testowanie na danych testowych (nie walidacyjnych)Unet-test
            self.trainer.test_model(test_gen,test_dmg_segmentation)
            dfs, df = self.trainer.compute_gen_measures(test_gen,self.class_weights, self.class_names)
            excel_path = self.trainer.create_model_dir('ExcelResults')
            with pd.ExcelWriter(os.path.join(excel_path,self.TASK_NAME+'_metrics.xlsx'), engine='openpyxl') as writer:
                 dfs.to_excel(writer, sheet_name='ICSHM', index=False)
                 df.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)

            dfs_tr, df_tr = self.trainer.compute_gen_measures(train_gen, self.class_weights, self.class_names)
            dfs_v, df_v = self.trainer.compute_gen_measures(validation_gen, self.class_weights, self.class_names)
            dfs_ts, df_ts = self.trainer.compute_gen_measures(test_gen, self.class_weights, self.class_names)
            excel_path = self.trainer.create_model_dir('ExcelResults')
            with pd.ExcelWriter(os.path.join(excel_path, self.TASK_NAME + '_all_sets_metrics.xlsx'),
                                engine='openpyxl') as writer:
                dfs_ts.to_excel(writer, sheet_name='ICSHM', index=False)
                df_ts.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)
                df_tr.to_excel(writer, sheet_name='ICSHM', index=False, startrow=20, startcol=0)
                df_v.to_excel(writer, sheet_name='ICSHM', index=False, startrow=30, startcol=0)
            self.trainer.predict('/home/piotrek/Computations/Ai/ICSHM/Photos/PredictionPhotos',write_prediction_segmentated2)
        else:
            print('Folder ',self.TASK_NAME,' exists. Model not trained')

        return ~self.trainer.model_dir_exists

    def compute_all_sets_measures(self):
        self.dataSource = DataSource(self.TRAIN_PATH, train_ratio=0.70, validation_ratio=0.15 )
        self.trainer = DLTrainer(self.TASK_PATH, self.TASK_NAME, self.model)
        if self.trainer.model_dir_exists:
            train_set, validation_set = self.dataSource.get_training_data()
            train_gen = DataGeneratorFromNumpyFiles(train_set, self.BATCH_SIZE, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES, augmentation_fn=self.augmentation_fn)
            validation_gen = DataGeneratorFromNumpyFiles(validation_set, 1, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES,shuffle=False)
            test_gen = DataGeneratorFromNumpyFiles(self.dataSource.get_test_files(), 1, (self.RES_Y, self.RES_X), (self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES,shuffle=False)
            model = self.trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

            dfs_tr, df_tr = self.trainer.compute_gen_measures(train_gen, self.class_weights, self.class_names)
            dfs_v, df_v = self.trainer.compute_gen_measures(validation_gen, self.class_weights, self.class_names)
            dfs_ts, df_ts = self.trainer.compute_gen_measures(test_gen,self.class_weights, self.class_names)
            excel_path = self.trainer.create_model_dir('ExcelResults')
            with pd.ExcelWriter(os.path.join(excel_path,self.TASK_NAME+'_all_sets_metrics.xlsx'), engine='openpyxl') as writer:
                 dfs_ts.to_excel(writer, sheet_name='ICSHM', index=False)
                 df_ts.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)
                 df_tr.to_excel(writer, sheet_name='ICSHM', index=False, startrow=20, startcol=0)
                 df_v.to_excel(writer, sheet_name='ICSHM', index=False, startrow=30, startcol=0)

        else:
            print('Folder ',self.TASK_NAME,' not exists. Model not trained')

        return ~self.trainer.model_dir_exists

    def predict(self, img_source, postprocess):
        index=1
        self.trainer = DLTrainer(self.TASK_PATH, self.TASK_NAME, self.model)
        print('Predicting images from dir:', img_source)
        N = len(os.listdir(img_source))
        for filename in os.listdir(img_source):
            try:
                #data_x = inputImgReader(os.path.join(img_source, filename))
                data_x = cv.resize(cv.imread(os.path.join(img_source, filename), (self.resY, self.resX), anti_aliasing=True)).astype('float32')
                data_y = self.trainer.model.predict(np.expand_dims(data_x,0))
                postprocess(os.path.join(self.predictions_path, filename), data_x, data_y[0,])
                # cv.imwrite(os.path.join(prediction_path, filename) + '_X.png',data_x*255 )
                # cv.imwrite(os.path.join(prediction_path, filename) + '_PRED.png', postprocess(data_x, data_y[0,]) * 255)
            except Exception as e:
                print('Cant import ' + filename + ' because', e)
            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)



class ICSHM_structural_task(ICSHM_Task):
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, TRAIN_DIR = 'Struct',RES_X=640, RES_Y=320, BATCH_SIZE=32, LEARNING_RATE = 0.00005, augmentation_fn=None):
        super().__init__(model=model,TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, N_CLASSES=4, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, augmentation_fn=augmentation_fn)
        self.class_weights = np.array([0.07, 0.33, 0.35, 0.25])
        self.csv_ind=5
        self.class_names = [ "Nonstructural", "Slab", "Beam", "Column" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset(os.path.join('TrainSets',TRAIN_DIR),ICSHM_STRUCT_Converter(self.RES_X, self.RES_Y))

class ICSHM_structural_depth_task(ICSHM_Task):
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, TRAIN_DIR = 'StructD',RES_X=640, RES_Y=320, BATCH_SIZE=32, LEARNING_RATE = 0.00005, augmentation_fn=None):
        super().__init__(model=model,TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, N_CHANNELS=4, N_CLASSES=4, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, augmentation_fn=augmentation_fn)
        self.class_weights = np.array([0.05, 0.3, 0.65])
        self.csv_ind=5
        self.class_names = [ "Nonstructural", "Slab", "Beam", "Column" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset(os.path.join('TrainSets',TRAIN_DIR),ICSHM_STRUCTD_Converter(self.RES_X, self.RES_Y))



class ICSHM_damage_task(ICSHM_Task):
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, TRAIN_DIR = 'Dmg', RES_X=640, RES_Y=320, BATCH_SIZE=32, LEARNING_RATE = 0.00005, augmentation_fn=None):
        super().__init__(model=model,TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, N_CLASSES=3,BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, augmentation_fn=augmentation_fn)
        #self.class_weights = np.array([ 0.00174144, 0.09980335, 0.8984552 ])
        self.class_weights = np.array([0.0, 0.6, 0.4])
        self.csv_ind = 6
        self.class_names = [ "Background", "Cracks", "Reinforcement" ]
        #self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        #self.loss_fn = weighted_tversky_loss(self.class_weights / np.sum(self.class_weights))
        self.loss_fn = weighted_focal_tversky_loss(self.class_weights / np.sum(self.class_weights))
        #self.loss_fn = tf.keras.losses.Dice()
        #self.loss_fn = wrapped_tversky_loss
        #self.loss_fn = focal_tversky_loss
        #self.loss_fn = CategoricalFocalCrossentropy(gamma=2.0, from_logits=False)
        self.create_dataset(os.path.join('TrainSets',TRAIN_DIR),ICSHM_DMG_Converter(self.RES_X, self.RES_Y))


def multi_augmentation_training_structural(model_basename, create_model_fn, task_fn, BATCH_SIZE, augmentations  ):
    tf.keras.backend.clear_session()
    print("* MULTI augmented training for model :",model_basename )
    for augmentation in augmentations:
        model = create_model_fn()
        task = task_fn( model_basename, model, augmentation, BATCH_SIZE)
        task.train()
        del model
        gc.collect()


def multi_augmentation_transfer_learning( model_basename, create_model_fn, task_fn, BATCH_SIZE, augmentations):
    tf.keras.backend.clear_session()
    print("* MULTI augmented transfer learning for model :", model_basename)
    for augmentation in augmentations:
        model, backbone = create_model_fn()
        for layer in backbone.layers:
            layer.trainable = False
        task = task_fn(model_basename + '_TR_' + augmentation[1], model, augmentations, BATCH_SIZE, LEARNING_RATE=0.001)
        task.train()
        for layer in backbone.layers:
            layer.trainable = True
        task = task_fn(model_basename + '_FT_' + augmentation[1], model, augmentations, BATCH_SIZE, LEARNING_RATE=0.0000045)
        task.train()
        del model
        gc.collect()

def predict_photos_in_all_tasks(task_path,task_name, photos_test_path,resX,resY):
    print('Photo prediction in task :',task_name)
    path_name= os.path.join(task_path,task_name)
    trainer = DLTrainer(task_path, task_name)
    prediction_photos_dir = os.path.join(path_name,'PhotoPredictions')
    if not os.path.exists(prediction_photos_dir):
        os.mkdir(prediction_photos_dir)
    trainer.predict(photos_test_path,write_prediction_segmentated3,resX,resY,prediction_photos_dir)

def compute_measures(task_path, task_name, photos_numpy_test_path, resX, resY, weights, class_names):
    print('Photos measures in task :', task_name)
    path_name = os.path.join(task_path, task_name)
    npx_files = [os.path.join(photos_numpy_test_path, name) for name in os.listdir(photos_numpy_test_path)]
    nclasses = len(weights)
    test_gen = DataGeneratorFromNumpyFiles(npx_files, 1, (resY, resX),
                                           (resY, resX), 3, nclasses, shuffle=False)
    trainer = DLTrainer(task_path, task_name)
    dfs, df = trainer.compute_gen_measures(test_gen, np.array(weights), class_names)
    prediction_photos_dir = os.path.join(path_name, 'PhotoPredictions')
    if not os.path.exists(prediction_photos_dir):
        os.mkdir(prediction_photos_dir)
    with pd.ExcelWriter(os.path.join(prediction_photos_dir, task_name + '_metrics.xlsx'), engine='openpyxl') as writer:
        dfs.to_excel(writer, sheet_name='ICSHM', index=False)
        df.to_excel(writer, sheet_name='ICSHM', index=False, startrow=10, startcol=0)

def compute_all_sets_measures(task_path, task_name, photos_numpy_test_path, resX, resY, weights, class_names):
    print('Photos measures in task :', task_name)
    path_name = os.path.join(task_path, task_name)
    trainer = DLTrainer(task_path, task_name)



    # model = create_model_fn()
    # task = task_fn(model_basename + "_br", model, augment_brightness, BATCH_SIZE )
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_cn", model, augment_contrast, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_gm", model, augment_gamma, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_ns", model, augment_noise, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_fl", model, augment_flip, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_rot", model, augment_rotation, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_cut", model, augment_cutmix, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()
    #
    # model = create_model_fn()
    # task = task_fn(model_basename + "_all", model, augment_all, BATCH_SIZE)
    # task.train()
    # del model
    # gc.collect()


