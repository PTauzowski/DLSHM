import os.path
import glob
import numpy as np
import tensorflow as tf
import random
import cv2 as cv

from dlshm.dlimages.convert import ImageTrainingPairMultiscaleAugmented,RandomSetParameter

def gener_test(pathName, data_generator, scope=-1):
    N=scope
    if N==-1:
        N=len(data_generator)
    k=0;
    for data_x, data_y in data_generator:
        for b in range(0,data_x.shape[0]):
            cv.imwrite( os.path.join(pathName, f"input_{k}_{b}.png"), data_x[b,] * 255)
            cv.imwrite( os.path.join(pathName, f"output_{k}_{b}.png"), data_y[b,] * 255)
        k+=1
        if k>N:
            break

def sequenced_gener_test(pathName, data_generator, scope=-1):
    N=scope
    if N==-1:
        N=len(data_generator)
    k=0;
    for data_x, data_y in data_generator:
        for b in range(0,data_x.shape[0]):
            for i in range(0, data_x.shape[1]):
                cv.imwrite( os.path.join(pathName, f"input_{k}_{b}_{i}.png"), data_x[b,i,] * 255)
                cv.imwrite( os.path.join(pathName, f"output_{k}_{b}_{i}.png"), data_y[b,i] * 255)
        k+=1
        if k>N:
            break

class DataSource:
    pass
    def __init__(self, sourceDir, train_ratio=0.8, validation_ratio=0.15, sampleSize=-1, shuffle=True, cross_validation_folds=1):
        self.sourceDir=sourceDir
        self.trainRatio=train_ratio
        self.validation_ratio=validation_ratio
        self.shuffle=shuffle
        self.sampleSize=sampleSize
        self.CROSS_VALIDATION_FOLDS=cross_validation_folds
        if cross_validation_folds>1:
            self.test_ratio = 1-train_ratio-validation_ratio
            self.validation_ratio=train_ratio/cross_validation_folds

        if self.sourceDir == "":
            print("Source path can't be empty")
            return
        if not os.path.isdir(self.sourceDir):
            print("Source path :", self.sourceDir + "Is not valid directory name. Processing aborted.")
            return
        print('Reading images from ', self.sourceDir)
        self.files = glob.glob(self.sourceDir + '/*')
        if shuffle:
            random.shuffle(self.files)
        print('Number of images :', len(self.files))
        if  len(self.files)==0:
            print("Source dir can't be empty")
            return
        self.total_sample_size = len(self.files)
        if self.sampleSize < 0:
            self.used_sample_size = self.total_sample_size
        else:
            self.used_sample_size = self.sampleSize

        self.train_samples_size = int(round(self.used_sample_size * self.trainRatio))
        self.validation_samples_size = int(round(self.used_sample_size * self.validation_ratio))
        self.test_samples_size = self.used_sample_size - self.train_samples_size - self.validation_samples_size

    def get_train_set_files(self):

            return self.files[:self.train_samples_size]
    def get_validation_set_files(self):
        return self.files[self.train_samples_size:self.train_samples_size+self.validation_samples_size]

    def get_test_set_files(self):
        return self.files[self.train_samples_size+self.validation_samples_size:self.used_sample_size]

    def get_dims(self):
        with open(self.files[0], 'rb') as f:
            x = np.load(f)
            y = np.load(f)
            return x.shape, y.shape

    def print_info(self):
        print('Data path :',self.sourceDir)
        print('Sample size :', self.total_sample_size)
        print('Used sample size :', self.used_sample_size)
        print('Training set size :',self.train_samples_size)
        print('Validation set size :', self.validation_samples_size)
        print('Test set size :', self.test_samples_size)

class DataGeneratorFromImageFilesAugmented(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, idim=(32, 32), odim=(32, 32), n_channels=3,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.idim = idim
        self.odim = odim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.ranfomFiles=RandomSetParameter(files)
        self.imageObj = ImageTrainingPairMultiscaleAugmented(idim[0],idim[1],odim[0]//idim[0])
        self.data_X = np.empty((self.batch_size, *self.idim, self.n_channels), dtype=np.float32)
        self.data_Y = np.empty((self.batch_size, *self.odim, self.n_classes), dtype=np.float32)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files*5) / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.ranfomFiles.get() for k in indexes]

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
        for i, ID in enumerate(list_IDs_temp):
            self.data_X[i,], self.data_Y[i,] = self.imageObj.get_images(cv.imread(ID, cv.IMREAD_UNCHANGED) / 255.0)

        return self.data_X, self.data_Y


class DataGeneratorFromNumpyFiles(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, idim=(32, 32), odim=(32, 32), n_channels=3,
                 n_classes=3, shuffle=True, Augmentation=False):
        'Initialization'
        self.idim = idim
        self.odim = odim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.shuffle = shuffle
        self.Augmentation=Augmentation
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

    def augment_random_image_transformation(X, Y):

        """Random flipping of the image (both vertical and horisontal) taking care on the seed - mask remains consistent with corresponding training image"""

        rnI = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        if rnI == 1:
            Xout = tf.image.flip_left_right(X)
            Yout = tf.image.flip_left_right(Y)
        elif rnI == 2:
            Xout = tf.image.flip_up_down(X)
            Yout = tf.image.flip_up_down(Y)
        elif rnI > 2:
            angle = np.random.uniform(-30, 30)
            (Xh, Xw) = X.shape[:2]
            Xcenter = (Xw // 2, Xh // 2)
            (Yh, Yw) = Y.shape[:2]
            Ycenter = (Yw // 2, Yh // 2)
            MX = cv.getRotationMatrix2D(Xcenter, np.int32(angle), 1.0)
            MY = cv.getRotationMatrix2D(Ycenter, np.int32(angle), 1.0)
            Xout = cv.warpAffine(X, MX, (Xw, Xh))
            Yout = cv.warpAffine(Y, MY, (Yw, Yh))
        return Xout, Yout
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        X = np.empty((self.batch_size, *self.idim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.odim, self.n_classes), dtype=np.float32)
        for i, ID in enumerate(list_IDs_temp):
            with open( ID, 'rb') as f:
                X[i,] = np.load(f)
                Y[i,] = np.load(f)

                if self.Augmentation:   # Data augmentation
                    # Quality changing (NOT applied to ground truth data):
                    X[i,] = tf.image.random_brightness(X[i,], max_delta=0.8).numpy()  # Random brightness
                    X[i,] = tf.image.random_contrast(X[i,], lower=0.1, upper=1.9).numpy()  # Random contrast
                    # Transformations, e.g. rotation, shifting (applied also to the ground truth data):
                    #X[i,], Y[i,] = augment_random_image_transformation(X[i,], Y[i,])
        
        return X, Y

class DataGeneratorFromNumpyFilesMem(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, idim=(32, 32), odim=(32, 32), n_channels=3,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.idim = idim
        self.odim = odim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.X = np.empty((len(files), *self.idim, self.n_channels), dtype=np.float32)
        self.Y = np.empty((len(files), *self.odim, self.n_classes), dtype=np.float32)
        for i, ID in enumerate(files):
            with open( ID, 'rb') as f:
                self.X[i,] = np.load(f)
                self.Y[i,] = np.load(f)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        return self.X[indexes,], self.Y[indexes,]

class DataGeneratorHalfSequences(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batch_size=32, idim=(32, 32), odim=(32, 32), n_channels=3,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = idim
        self.odim = odim
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
        #     np.zeros((resX // 2 - 1, 3, resY, resX // 2), dtype=np.float32)
        X = np.empty((self.batch_size, self.dim[1] // 2 - 1, self.dim[0], self.dim[1] // 2, 3), dtype=np.float32)
        Y = np.empty((self.batch_size, self.dim[1] // 2 - 1, self.dim[0], self.dim[1] // 2, 3), dtype=np.float32)
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