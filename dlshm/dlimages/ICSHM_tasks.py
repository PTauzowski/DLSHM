import os
import gc
import numpy as np
import tensorflow as tf
import pandas as pd

from dlshm.dlgenerators.generators import DataSource, DataGeneratorFromNumpyFiles
from dlshm.dlimages.augmentations import augment_brightness, augment_contrast, augment_gamma, augment_noise, \
    augment_rotation, augment_cutmix, augment_all, augment_flip
from dlshm.dlimages.postprocess import test_dmg_segmentation, write_prediction_segmentated2
from dlshm.dlmodels import trainer
from dlshm.dlmodels.loss_functions import weighted_categorical_crossentropy
from dlshm.dlimages.data_processing import ICSHM_STRUCT_Converter, ICSHM_DMG_Converter, ICSHMDataManager
from dlshm.dlmodels.trainer import DLTrainer


class ICSHM_Task:
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, N_CHANNELS=3, N_CLASSES=4, N_LAYERS=6, N_FILTERS=24, BATCH_SIZE=32, EPOCHS=200, LEARNING_RATE = 0.001, augmentation_fn=None):
        self.model=model
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

    def create_dataset(self,train_dir,converter):
        self.data_manager = ICSHMDataManager(self.SOURCE_PATH,csv_ind=self.csv_ind)
        self.TRAIN_PATH = os.path.join(self.TASK_PATH, train_dir)
        self.data_manager.convert_data_to_numpy_format(converter, self.TRAIN_PATH)

    def train(self):
        self.dataSource = DataSource(self.TRAIN_PATH, train_ratio=0.75, validation_ratio=0.15 )
        self.trainer = DLTrainer(self.TASK_PATH, self.TASK_NAME, self.model)
        train_set, validation_set = self.dataSource.get_training_data()
        train_gen = DataGeneratorFromNumpyFiles(train_set, self.BATCH_SIZE, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES, augmentation_fn=self.augmentation_fn)
        validation_gen = DataGeneratorFromNumpyFiles(validation_set, 1, (self.RES_Y, self.RES_X),(self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES)
        test_gen = DataGeneratorFromNumpyFiles(self.dataSource.get_test_files(), 1, (self.RES_Y, self.RES_X), (self.RES_Y, self.RES_X), self.N_CHANNELS, self.N_CLASSES)
        model = self.trainer.model  # Gdyby model powyżej nie był podany ("none" - jak w komentarzu), to tutaj go "wydobywamy"

        # Kompilacja modelu i wyswitlenie informacji:
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(N_CLASSES)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss="categorical_crossentropy",
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanIoU(self.N_CLASSES)])
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
        self.trainer.predict('/home/piotrek/Computations/Ai/ICSHM/Photos/PredictionPhotos',write_prediction_segmentated2)



class ICSHM_structural_task(ICSHM_Task):
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, BATCH_SIZE=32 , augmentation_fn=None):
        super().__init__(model=model,TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, N_CLASSES=4,BATCH_SIZE=BATCH_SIZE,augmentation_fn=augmentation_fn)
        self.class_weights = np.array([0.07, 0.33, 0.35, 0.25])
        self.csv_ind=5;
        self.class_names = [ "Nonstructural", "Slab", "Beam", "Column" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset(os.path.join('TrainSets','Struct'),ICSHM_STRUCT_Converter(self.RES_X, self.RES_Y))


class ICSHM_damage_task(ICSHM_Task):
    def __init__(self, model, TASK_PATH, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, BATCH_SIZE=32, augmentation_fn=None):
        super().__init__(model=model,TASK_PATH=TASK_PATH, SOURCE_PATH=SOURCE_PATH, TASK_NAME=TASK_NAME, RES_X=RES_X, RES_Y=RES_Y, N_CLASSES=3,BATCH_SIZE=BATCH_SIZE, augmentation_fn=augmentation_fn)
        self.class_weights = np.array([ 0.00174144, 0.09980335, 0.8984552 ])
        self.csv_ind = 6;
        self.class_names = [ "Background", "Cracks", "Reinforcement" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset(os.path.join('TrainSets','Dmg'),ICSHM_DMG_Converter(self.RES_X, self.RES_Y))


def multi_augmentation_training_structural(model_basename, create_model_fn, task_fn, BATCH_SIZE, augmentations  ):
    tf.keras.backend.clear_session()
    print("* MULTI augmented training for model :",model_basename )
    for augmentation in augmentations:
        model = create_model_fn()
        task = task_fn( model_basename + augmentation.postfix, model, augmentation, BATCH_SIZE)
        task.train()
        del model
        gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_br", model, augment_brightness(), BATCH_SIZE )
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_cn", model, augment_contrast, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_gm", model, augment_gamma, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_ns", model, augment_noise, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_fl", model, augment_flip, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_rot", model, augment_rotation, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_cut", model, augment_cutmix, BATCH_SIZE)
    task.train()
    del model
    gc.collect()

    model = create_model_fn()
    task = task_fn(model_basename + "_all", model, augment_all, BATCH_SIZE)
    task.train()
    del model
    gc.collect()


