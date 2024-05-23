import time
import os
import matplotlib
import cv2 as cv
import tensorflow as tf
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
import keras.backend as K

matplotlib.use('Agg')

def psnr(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=1)[0]
    return psnr_value

class DLTrainer:
    def __init__(self,model_name, model, task_path, data_source, trainGen, validGen, testGen, batch_size, idim, odim, n_channels, n_classes ):
        self.task_path=task_path
        self.model_name=model_name
        if model==None:
            model_path = os.path.join(self.task_path, 'Models')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(os.path.join(model_path, self.model_name+'.h5'),compile=False)
                print('Model ',model_name,'was found and loaded')
            else:
                print('Model ', model_name, 'was NOT found')
        self.model=model
        self.data_source=data_source
        self.idim=idim
        self.odim=odim
        self.n_channels=n_channels
        self.n_class=n_classes
        #self.train_gen = trainGen(data_source.getTrainSetFiles(), batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes,  class_weights=class_weights, shuffle=True )
        #self.validation_gen = validGen(data_source.getValidationSetFiles(), batch_size=batch_size, dim=dim, n_channels=n_channels,n_classes=n_classes, class_weights=class_weights, shuffle=True )
        #self.testGen = testGen(data_source.getTestSetFiles(),batch_size=1, dim=dim, n_channels=n_channels,n_classes=n_classes, class_weights=class_weights, shuffle=True )
        if data_source != None:
            self.train_gen = trainGen(data_source.get_train_set_files(), batch_size=batch_size, idim=idim, odim=odim, n_channels=n_channels, n_classes=n_classes, shuffle=True)
            self.validation_gen = validGen(data_source.get_validation_set_files(), batch_size=batch_size, idim=idim, odim=odim, n_channels=n_channels, n_classes=n_classes, shuffle=True)
            self.testGen = testGen(data_source.get_test_set_files(), batch_size=1, idim=idim, odim=odim, n_channels=n_channels, n_classes=n_classes, shuffle=True)

    def save_model(self):
        model_path = os.path.join(self.task_path, 'Models')
        isExist = os.path.exists(model_path)
        if not isExist:
            os.makedirs(model_path)
        self.model.save(os.path.join(model_path, self.model_name)+'.h5')

    def train(self,epochs,batch_size):
        training_time_start = time.process_time()
        self.history = self.model.fit(self.train_gen, batch_size=batch_size, epochs=epochs, validation_data=self.validation_gen)
        self.save_model()
        self.training_time=time.process_time() - training_time_start

    def plot_training_history(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        plt.figure()
        plt.plot(self.history.epoch, loss, 'r', label='Training loss')
        plt.plot(self.history.epoch, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        model_path = os.path.join(self.task_path, 'Models')
        plt.savefig(os.path.join(model_path, self.model_name)+'_training.png')

    def plot_training_accuracy(self):
        fig, axis = plt.subplots(1, 2, figsize=(20, 5))
        axis[0].plot(self.history.history.history["loss"], color='r', label='train loss')
        axis[0].plot(self.history.history.history["val_loss"], color='b', label='dev loss')
        axis[0].set_title('Loss Comparison')
        axis[0].legend()
        axis[1].plot(self.history.history.history["accuracy"], color='r', label='train accuracy')
        axis[1].plot(self.history.history.history["val_accuracy"], color='b', label='dev accuracy')
        axis[1].set_title('Accuracy Comparison')
        axis[1].legend()

    def test_model_images(self, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(self.testGen)
        print('Testing model:',self.model_name)
        for x,y in self.testGen:
            data_y = self.model.predict(x)
            cv.imwrite(os.path.join(test_path, 'test_file_T_' + str(index))+'.'+extension, postprocess(x[0,],data_y[0,])*255)
            cv.imwrite(os.path.join(test_path, 'test_file_Y_' + str(index)) + '.' + extension, y[0,]*255)
            cv.imwrite(os.path.join(test_path, 'test_file_X_' + str(index)) + '.' + extension, x[0,]*255)
            index=index+1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)

    def test_model(self, postprocess, extension='png'):
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


    def test_model_weighted(self, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(self.testGen)
        print('Testing model:',self.model_name)
        for x,y,weights in self.testGen:
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
                cv.imwrite(os.path.join(prediction_path, filename) + '_X.png',data_x*255 )
                cv.imwrite(os.path.join(prediction_path, filename) + '_PRED.png', postprocess(data_x, data_y[0,]) * 255)
            except Exception as e:
                print('Cant import ' + filename + ' because', e)
            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)

    def compute_measures(self, img_source, inputImgReader, postprocess):
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
                true_y = cv.imread(os.path.join(img_source, 'label_'+filename))
                data_y = self.model.predict(np.expand_dims(data_x,0))

                threshold = 0.5


                mask = np.empty((320, 640, 8), dtype=np.float32)
                mask[:, :, 0] = np.where(true_y[:,:,1] == 1, 1, 0)
                mask[:, :, 1] = np.where(true_y[:,:,1] == 2, 1, 0)
                mask[:, :, 2] = np.where(true_y[:,:,1] == 3, 1, 0)
                mask[:, :, 3] = np.where(true_y[:,:,1] == 4, 1, 0)

                accuracy = np.sum(mask[:, :, 1]  * data_y[0,:,:,1])/np.sum(mask[:, :, 2] )

                # correct_predictions = np.sum(pred_mask == true_mask)
                # total_pixels = pred_mask.size
                # accuracy = correct_predictions / total_pixels

                idx_pred=2
                idx_mask=0

                cce_loss = tf.keras.losses.CategoricalCrossentropy()
                cce_value = cce_loss(mask[:, :, idx_mask], data_y[0, :, :, idx_pred])

                mean_iou_fn = tf.keras.metrics.MeanIoU(num_classes=1)
                mean_iou_fn.update_state(mask[:, :, idx_mask], data_y[0, :, :, idx_pred])
                keras_iou=mean_iou_fn.result().numpy()



                accuracy_fn = tf.keras.metrics.Accuracy()
                keras_accuracy = accuracy_fn(mask[:, :, idx_mask], data_y[0, :, :, idx_pred])


                binary_mask = (mask[:, :, idx_mask] >= threshold).astype(int)
                binary_pred = (data_y[0, :, :, idx_pred] >= threshold).astype(int)
                TP = np.sum((binary_pred == 1) & (binary_mask == 1))
                FP = np.sum((binary_pred == 1) & (binary_mask == 0))
                FN = np.sum((binary_pred == 0) & (binary_mask == 1))
                TN = np.sum((binary_pred == 0) & (binary_mask == 0))

                correct_predictions = np.sum(mask[:, :, idx_mask] * data_y[0, :, :, idx_pred])
                correct_negative_predictions = np.sum((1-mask[:, :, idx_mask]) * (1-data_y[0, :, :, idx_pred]))
                false_negative_predictions = np.sum((1 - mask[:, :, idx_mask]) * data_y[0, :, :, idx_pred])
                false_positive_predictions = np.sum(mask[:, :, idx_mask] * (1 - data_y[0, :, :, idx_pred]))
                total_pixels = binary_pred.size
                accuracy = (correct_predictions + correct_negative_predictions ) / total_pixels
                iou = correct_predictions / (correct_predictions + false_positive_predictions + false_negative_predictions )

                #intersection = np.sum(mask[:, :, idx_mask] * data_y[0, :, :, idx_pred])

                # true positive / (true positive + false positive + false negative)

                intersection = np.sum(data_y[0, :, :, idx_pred] * mask[:, :, idx_mask])
                union = np.sum(np.maximum(binary_mask, binary_pred))
                iou = intersection / union # if union != 0 else 0  # To
                # Calculate IoU
                bin_iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0  # To avoid division by zero

                # Calculate accuracy
                bin_accuracy = (TP + TN) / (TP + FP + FN + TN)  # All pixels

                print("IoU:", bin_iou)
                print("Accuracy:", bin_accuracy)

            except Exception as e:
                print('Cant import ' + filename + ' because', e)
            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)

    def mosaic_predict(self, img_source_dir, idim, odim):
        predictions_dir = os.path.join(self.task_path, 'mosaicPredictions')
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        prediction_path = os.path.join(predictions_dir, self.model_name)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        index = 1
        print('Predicting images from dir:', img_source_dir)
        N = len(os.listdir(img_source_dir))
        for filename in os.listdir(img_source_dir):
            image= cv.imread(os.path.join(img_source_dir,filename))
            nx = image.shape[1] // idim[0]
            ny = image.shape[0] // idim[1]
            x0 = (image.shape[1] - idim[0] * nx) // 2
            y0 = (image.shape[0] - idim[1] * ny) // 2
            data_x = np.zeros((1,idim[1],idim[0],3),dtype=np.float32)
            image_y = np.zeros((odim[1]*ny, odim[0]*nx,3),dtype=np.float32)
            for l in range(0, ny):
                for k in range(0, nx):
                    data_x[0,] = image[y0+l*idim[1]:y0+(l+1)*idim[1],x0+k*idim[0]:x0+(k+1)*idim[0],:]/255
                    data_y = self.model.predict(data_x)
                    image_y[l*odim[1]:(l+1)*odim[1], k*odim[0]:(k+1)*odim[0], :] = data_y[0,]
            try:
                cv.imwrite(os.path.join(prediction_path, filename) + '_X.png', image)
                cv.imwrite(os.path.join(prediction_path, filename) + '_PRED.png', image_y * 255)
            except Exception as e:
                print('Cant export ' + filename + ' because', e)
            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)



