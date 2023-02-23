import os.path
import glob
import numpy as np
import tensorflow as tf
import random

class DataSource:
    def __init__(self, sourceDir, trainRatio=0.8, validationRatio=0.15, sampleSize=-1, shuffle=True ):
        self.sourceDir=sourceDir
        self.trainRatio=trainRatio
        self.validationRatio=validationRatio
        self.shuffle=shuffle
        self.sampleSize = sampleSize
        self.initData()

    def initData(self):
        if self.sourceDir == "":
            print("Source path can't be empty")
            return False
        if not os.path.isdir(self.sourceDir):
            print("Source path :", self.sourceDir + "Is not valid directory name. Processing aborted.")
            return False
        print('Reading images from ', self.sourceDir)
        self.files = glob.glob(self.sourceDir + '\\*.*')
        random.shuffle(self.files)
        print('Number of images :', len(self.files))
        if  len(self.files)==0:
            print("Source dir can't be empty")
            return False
        self.total_sample_size = len(self.files)
        if self.sampleSize < 0:
            self.used_sample_size = self.total_sample_size
        else:
            self.used_sample_size = self.sampleSize
        self.train_samples_size = int(round(self.used_sample_size * self.trainRatio))
        self.validation_samples_size = int(round(self.used_sample_size * self.validationRatio))
        self.test_samples_size = self.used_sample_size - self.train_samples_size - self.validation_samples_size
        return  True

    def getTrainSetFiles(self):
        return self.files[:self.train_samples_size]

    def getValidationSetFiles(self):
        return self.files[self.train_samples_size:self.train_samples_size+self.validation_samples_size]

    def getTestSetFiles(self):
        return self.files[self.train_samples_size+self.validation_samples_size:self.used_sample_size]

    def getDims(self):
        with open(self.files[0], 'rb') as f:
            x = np.load(f)
            y = np.load(f)
            return x.shape, y.shape

    def printInfo(self):
        print('Data path :',self.sourceDir)
        print('Sample size :', self.total_sample_size)
        print('Used sample size :', self.used_sample_size)
        print('Training set size :',self.train_samples_size)
        print('Validation set size :', self.validation_samples_size)
        print('Test set size :', self.test_samples_size)



class DataGeneratorFromNumpyFiles(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)
        for i, ID in enumerate(list_IDs_temp):
            with open( ID, 'rb') as f:
                X[i,] = np.load(f)
                Y[i,] = np.load(f)

        return X, Y

class DataGeneratorFromNumpyFilesWeighted(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, dim=(32, 32), n_channels=3,  n_classes=3, class_weights=[], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        class_weights = self.class_weights / tf.reduce_sum(self.class_weights)

        sample_weights = y[:, :, :, 0] * class_weights[0]
        for k in range(1,self.n_classes):
            sample_weights = sample_weights + y[:, :, :, k] * class_weights[k]

        return X, y, sample_weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)
        for i, ID in enumerate(list_IDs_temp):
            with open( ID, 'rb') as f:
                X[i,] = np.load(f)
                Y[i,] = np.load(f)

        return X, Y


class DataGeneratorWeighted(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filenames, class_weights, batch_size=32, dim=(32,32),  n_channels=1, n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.filenames = filenames
        self.class_weights = class_weights
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        filenames_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(filenames_temp)

        class_weights = self.class_weights / tf.reduce_sum(self.class_weights)

        sample_weights = y[:,:,:,0]*class_weights[0] + y[:,:,:,1]*class_weights[1] #+y[:,:,:,2]*class_weights[2];

        return X, y, sample_weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        X = np.empty((self.batch_size,*self.dim, self.n_channels))
        Y = np.empty((self.batch_size,*self.dim, self.n_classes))
        for i, ID in enumerate(filenames_temp):
            with open(os.path.join(self.dir_pathname, ID), 'rb') as f:
                X[i,] = np.load(f)
                Y[i,] = np.load(f)
        return X, Y