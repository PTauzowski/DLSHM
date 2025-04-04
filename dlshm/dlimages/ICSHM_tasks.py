import os
import numpy as np
from dlshm.dlmodels.loss_functions import weighted_categorical_crossentropy
from dlshm.dlimages.data_processing import ICSHM_STRUCT_Converter, ICSHM_DMG_Converter, ICSHMDataManager

class ICSHM_Task:
    def __init__(self, TASK_DIR, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, N_CHANNELS=3, N_CLASSES=4, N_LAYERS=6, N_FILTERS=24, LEARNING_RATE = 0.001):
        self.TASK_DIR
        self.RES_X = RES_X
        self.RES_Y = RES_Y
        self.N_CHANNELS = N_CHANNELS
        self.N_CLASSES = N_CLASSES
        self.N_LAYERS = N_LAYERS
        self.N_FILTERS = N_FILTERS
        self.LEARNING_RATE = LEARNING_RATE
        self.TASK_DIR = TASK_DIR
        self.SOURCE_PATH = SOURCE_PATH
        self.TASK_NAME = TASK_NAME

    def create_dataset(self,train_dir,converter):
        img_conv = converter(self.RES_X, self.RES_Y)  # konwersja na pliki npy - jak sa, to juz tego nie robi
        self.data_manager = ICSHMDataManager(self.SOURCE_PATH)
        self.TRAIN_PATH = os.path.join(self.TASK_DIR, train_dir)
        self.data_manager.convert_data_to_numpy_format(img_conv, self.TRAIN_PATH)



class ICSHM_structural_task(ICSHM_Task):
    def __init__(self, TASK_DIR, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, N_CHANNELS=3, N_CLASSES=4, N_LAYERS=6, N_FILTERS=24, LEARNING_RATE = 0.001):
        super().__init__(TASK_DIR, SOURCE_PATH, TASK_NAME, RES_X, RES_Y, N_CHANNELS, N_CLASSES, N_LAYERS, N_FILTERS, LEARNING_RATE)
        self.class_weights = np.array([0.07, 0.33, 0.35, 0.25])
        self.class_names = [ "Nonstructural", "Slab", "Beam", "Column" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset('TrainSetStruct',ICSHM_STRUCT_Converter(self.RES_X, self.RES_Y))



class ICSHM_damage_task(ICSHM_Task):
    def __init__(self, TASK_DIR, SOURCE_PATH, TASK_NAME, RES_X=640, RES_Y=320, N_CHANNELS=3, N_CLASSES=4, N_LAYERS=6,
                 N_FILTERS=24, LEARNING_RATE=0.001):
        super().__init__(TASK_DIR, SOURCE_PATH, TASK_NAME, RES_X, RES_Y, N_CHANNELS, N_CLASSES, N_LAYERS, N_FILTERS,
                         LEARNING_RATE)
        self.class_weights = np.array([ 0.00174144, 0.09980335, 0.8984552 ])
        self.class_names = [ "Background", "Cracks", "Reinforcement" ]
        self.loss_fn = weighted_categorical_crossentropy(self.class_weights / np.sum(self.class_weights))
        self.create_dataset('TrainSetDMg',ICSHM_DMG_Converter(self.RES_X, self.RES_Y))

