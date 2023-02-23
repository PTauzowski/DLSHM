import time
import os
import matplotlib
import cv2 as cv
import tensorflow as tf
import numpy as np
from matplotlib import image as mpimg, pyplot as plt

matplotlib.use('Agg')

class DLTrainer:
    def __init__(self,model_name, model, task_path, data_source, trainGen, validGen, testGen, batch_size, dim, n_channels, n_classes, class_weights ):
        self.task_path=task_path
        self.model_name=model_name
        if model==None:
            model_path = os.path.join(self.task_path, 'Models')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(os.path.join(model_path, self.model_name))
                print('Model ',model_name,'was found and loaded')
            else:
                print('Model ', model_name, 'was NOT found')
        self.model=model
        self.data_source=data_source
        self.dim=dim
        self.n_channels=n_channels
        self.n_class=n_classes
        self.train_gen = trainGen(data_source.getTrainSetFiles(), batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, class_weights=class_weights, shuffle=True, )
        self.validation_gen = validGen(data_source.getValidationSetFiles(), batch_size=batch_size, dim=dim, n_channels=n_channels,n_classes=n_classes, class_weights=class_weights, shuffle=True )
        self.testGen = testGen(data_source.getTestSetFiles(),batch_size=1, dim=dim, n_channels=n_channels,n_classes=n_classes, class_weights=class_weights, shuffle=True )

    def train(self,epochs,batch_size):
        training_time_start = time.process_time()
        # model = tf.keras.models.load_model(model_pathname)
        model_path = os.path.join(self.task_path, 'Models')
        isExist = os.path.exists(model_path)
        if not isExist:
            os.makedirs(model_path)
        self.history = self.model.fit(self.train_gen, batch_size=batch_size, epochs=epochs, validation_data=self.validation_gen)
        self.model.save(os.path.join(model_path, self.model_name))
        self.training_time=time.process_time() - training_time_start

    def plotTrainingHistory(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        plt.figure()
        plt.plot(self.history.epoch, loss, 'r', label='Training loss')
        plt.plot(self.history.epoch, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.show()

    def testModelImages(self, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(self.testGen)
        print('Testing model:',self.model_name)
        for x,y in self.testGen:
            data_y = self.model.predict(x)
            cv.imwrite(os.path.join(test_path, 'test_file_'+str(index))+'.'+extension, postprocess(x[0,],data_y[0,])*255)
            index=index+1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)

    def testModel(self, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(self.testGen)
        print('Testing model:',self.model_name)
        for x,y in self.testGen:
            data_y = self.model.predict(x)
            postprocess(os.path.join(test_path, 'test_file_'+str(index))+'.'+extension,x,y,data_y)
            index=index+1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)

    def predict(self,img_source, inputImgReader, postprocess):
        predictions_dir=os.path.join(self.task_path, 'Predictions')
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        prediction_path = os.path.join(predictions_dir, self.model_name)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        index=1
        print('Predicting images from dir:', img_source)
        N = len(os.listdir(img_source))
        for filename in os.listdir(img_source):
            try:
                data_x = inputImgReader(os.path.join(img_source, filename))
                data_y = self.model.predict(np.expand_dims(data_x,0))
                cv.imwrite(os.path.join(prediction_path, filename) + '.png', postprocess(data_x,data_y[0,])*255 )
            except Exception as e:
                print('Cant import ' + filename + ' because', e)
            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)




