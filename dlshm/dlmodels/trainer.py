import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
import tensorflow as tf

from tensorflow import keras
from flatbuffers.packer import float32
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# import datetime as dt

import numpy as np


def psnr(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=1)[0]
    return psnr_value


class DLTrainer:
    def __init__(self, model_name, model, task_path ):
        self.task_path=task_path
        self.model_name=model_name
        if model==None:
            model_path = os.path.join(self.task_path, 'Models')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(os.path.join(model_path, self.model_name+'.keras'),compile=False)
                print('Model ',model_name,'was found and loaded')
            else:
                print('Model ', model_name, 'was NOT found')
        self.model=model
        self.model_path = os.path.join(self.task_path, 'Models')
        isExist = os.path.exists(self.model_path)
        if not isExist:
            os.makedirs(self.model_path)
        self.model_filename=os.path.join(self.model_path, self.model_name)+'.keras'

    def save_model(self):
        model_path = os.path.join(self.task_path, 'Models')
        isExist = os.path.exists(model_path)
        if not isExist:
            os.makedirs(model_path)
        self.model.save(os.path.join(model_path, self.model_name)+'.keras')

    def train(self, train_gen, validation_gen, epochs, batch_size ):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_filename,  # Filepath to save the model
            monitor="val_loss",  # Metric to track (e.g., "val_accuracy" for classification)
            save_best_only=True,  # Save only when val_loss improves
            mode="min",  # "min" because lower loss is better
            verbose=1  # Print a message when saving
        )

        callbacks = [
            checkpoint_callback,
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
            EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
        ]

        training_time_start = time.process_time()
        self.history = self.model.fit(train_gen, batch_size=batch_size, epochs=epochs, validation_data=validation_gen, callbacks=callbacks)
        self.save_model()
        self.training_time=time.process_time() - training_time_start

    def test_model(self, test_gen, postprocess, extension='png'):
        test_path = os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        N=len(test_gen)
        print('Testing model:',self.model_name)
        for i, (x, y) in enumerate(test_gen):
            data_y = self.model.predict(x,verbose=0)
            postprocess(os.path.join(test_path, 'test_file_'+str(i))+'.'+extension,x,y,data_y)
            if i % 100 == 0:
                print('iter=', i, '/', N, flush=True)
            if i + 1 >= N:  # Stop after all batches
                break

    def test_model_on_images(self, test_gen, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(test_gen)
        print('Testing model:',self.model_name)
        for i, (x, y) in enumerate(test_gen):
            data_y = self.model.predict(x)
            cv.imwrite(os.path.join(test_path, 'test_file_T_' + str(index))+'.'+extension, postprocess(x[0,],data_y[0,])*255)
            cv.imwrite(os.path.join(test_path, 'test_file_Y_' + str(index)) + '.' + extension, y[0,]*255)
            cv.imwrite(os.path.join(test_path, 'test_file_X_' + str(index)) + '.' + extension, x[0,]*255)
            index=index+1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)
            if i + 1 >= N:  # Stop after all batches
                break


    def test_model_weighted(self, test_gen, postprocess, extension='png'):
        test_path=os.path.join(self.task_path, 'TestResults')
        test_path = os.path.join(test_path, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        index=1
        N=len(test_gen)
        print('Testing model:',self.model_name)
        for x,y,weights in test_gen:
            data_y = self.model.predict(x)
            postprocess(os.path.join(test_path, 'test_file_'+str(index))+'.'+extension,x,y,data_y)
            index=index+1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)
            if index + 1 >= N:  # Stop after all batches
                break

    def predict(self,img_source, postprocess):
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
                #data_x = inputImgReader(os.path.join(img_source, filename))
                data_x = cv.imread(os.path.join(img_source, filename)).astype('float32') / 255.0
                data_y = self.model.predict(np.expand_dims(data_x,0))
                postprocess(os.path.join(prediction_path, filename), data_x, data_y[0,])
                # cv.imwrite(os.path.join(prediction_path, filename) + '_X.png',data_x*255 )
                # cv.imwrite(os.path.join(prediction_path, filename) + '_PRED.png', postprocess(data_x, data_y[0,]) * 255)
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

    def compute_measures(self, data_y, true_y, weights):
        n_masks =  true_y.shape[2]
        accuracy_fn = tf.keras.metrics.Accuracy()
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
        accuracy  = np.zeros((n_masks), dtype=np.float32)
        keras_accuracy = np.zeros((n_masks), dtype=np.float32)
        cce_value = np.zeros((n_masks), dtype=np.float32)
        iou = np.zeros((n_masks), dtype=np.float32)
        precision = np.zeros((n_masks), dtype=np.float32)
        true_positives = np.zeros((n_masks), dtype=np.float32)
        true_negatives = np.zeros((n_masks), dtype=np.float32)
        false_negatives = np.zeros((n_masks), dtype=np.float32)
        false_positives = np.zeros((n_masks), dtype=np.float32)
        all_pixels = true_y[:,:,0].size
        epsilon=1.0E-7

        for k in range(0,n_masks):
            true_positives[k]   = np.sum(     true_y[:, :, k]  *      data_y[ :, :, k])
            true_negatives[k]   = np.sum((1 - true_y[:, :, k]) * (1 - data_y[ :, :, k]))
            false_positives[k]  = np.sum((1 - true_y[:, :, k]) *      data_y[ :, :, k])
            false_negatives[k]  = np.sum(     true_y[:, :, k]  * (1-  data_y[ :, :, k]))
            accuracy[k] = np.sum(true_y[:, :, k] * data_y[ :, :, k]) / np.sum(true_y[:, :, k])
            keras_accuracy[k] = accuracy_fn(true_y[:, :, k], data_y[ :, :, k])
            cce_value[k] = cce_loss(true_y[:, :, k], data_y[ :, :, k])

        precision = true_positives / (true_positives + false_positives+epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)
        dice_f1 = 2 * true_positives / (precision+recall + epsilon)
        dsc = 2 * true_positives / ( 2 * true_positives + false_positives + false_negatives + epsilon)
        specificity = true_negatives / ( true_negatives + false_positives + epsilon)
        balanced_accuracy = (recall + specificity) / 2
        iou = true_positives / (true_positives + false_positives + false_negatives + epsilon)
        m_iou = np.sum(iou) / n_masks
        global_iou = np.sum(true_positives) / (np.sum(true_positives+false_positives+false_negatives)+epsilon)
        return true_positives, true_negatives, false_positives, false_negatives, accuracy, keras_accuracy, cce_value, precision, recall, dice_f1, dsc, specificity, balanced_accuracy, iou, m_iou, global_iou

    def compute_binary_measures(self, data_y, true_y, weights):
        n_masks = true_y.shape[2]
        if weights.size != n_masks:
            weights = np.ones((n_masks),float32)

        true_positives = np.zeros((n_masks), dtype=int)
        true_negatives = np.zeros((n_masks), dtype=int)
        false_negatives = np.zeros((n_masks), dtype=int)
        false_positives = np.zeros((n_masks), dtype=int)

        for k in range(0,n_masks):
            true_positives[k] = np.sum((data_y[:, :, k] == 1) & (true_y[:, :, k] == 1))
            true_negatives[k] = np.sum((data_y[:, :, k] == 0) & (true_y[:, :, k] == 0))
            false_negatives[k] = np.sum((data_y[:, :, k] == 0) & (true_y[:, :, k] == 1))
            false_positives[k] = np.sum((data_y[:, :, k] == 1) & (true_y[:, :, k] == 0))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        dice_f1 = 2 * true_positives / (precision + recall)
        dsc = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        balanced_accuracy = (recall + specificity) / 2
        iou = true_positives / (true_positives + false_positives + false_negatives)
        m_iou = np.sum(iou) / n_masks
        global_iou = np.sum(true_positives) / np.sum(true_positives + false_positives + false_negatives)
        return true_positives, true_negatives, false_positives, false_negatives, accuracy, precision, recall, dice_f1, dsc, specificity, balanced_accuracy, iou, m_iou, global_iou

    def compute_gen_measures(self, sample_generator, weights, CLASS_NAMES):
        N = len(sample_generator)
        n_masks = weights.size
        epsilon = 1.0E-7
        index=0
        threshold = 0.5

        TP = np.zeros((n_masks), dtype=np.float64)
        TN = np.zeros((n_masks), dtype=np.float64)
        FN = np.zeros((n_masks), dtype=np.float64)
        FP = np.zeros((n_masks), dtype=np.float64)

        binTP = np.zeros((n_masks), dtype=np.uint64)
        binTN = np.zeros((n_masks), dtype=np.uint64)
        binFN = np.zeros((n_masks), dtype=np.uint64)
        binFP = np.zeros((n_masks), dtype=np.uint64)

        for i, (data_x, true_y) in enumerate(sample_generator):
            data_y = self.model.predict(data_x, verbose=0)
            binary_mask = (true_y >= threshold).astype(int)
            binary_pred = (data_y >= threshold).astype(int)

            for k in range(0, n_masks):
                TP[k] = TP[k] + np.sum(true_y[0, :, :, k] * data_y[0, :, :, k])
                TN[k] = TN[k] + np.sum((1 - true_y[0, :, :, k]) * (1 - data_y[0, :, :, k]))
                FP[k] = FP[k] + np.sum((1 - true_y[0, :, :, k]) * data_y[0, :, :, k])
                FN[k] = FN[k] + np.sum(true_y[0, :, :, k] * (1 - data_y[0, :, :, k]))
                binTP[k] = binTP[k] + np.sum((binary_pred[0, :, :, k] == 1) & (binary_mask[0, :, :, k] == 1))
                binTN[k] = binTN[k] + np.sum((binary_pred[0, :, :, k] == 0) & (binary_mask[0, :, :, k] == 0))
                binFN[k] = binFN[k] + np.sum((binary_pred[0, :, :, k] == 0) & (binary_mask[0, :, :, k] == 1))
                binFP[k] = binFP[k] + np.sum((binary_pred[0, :, :, k] == 1) & (binary_mask[0, :, :, k] == 0))

            index = index + 1
            if index % 100 == 0:
                print('iter=', index, '/', N, flush=True)
            if index + 1 >= N:  # Stop after all batches
                break
        class_accuracy = (TP + TN) / ( TP + TN + FP + FN + epsilon)
        cat_accuracy = np.sum((TP + TN) / ( TP + TN + FP + FN + epsilon) ) / n_masks
        global_accuracy = np.sum(TP + TN) / (np.sum(TP + TN + FP + FN) + epsilon)
        weighted_accuracy = np.sum( weights * (TP + TN) / ( TP + TN + FP + FN + epsilon) )

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        dice_f1 = 2 * TP / (precision + recall + epsilon)
        dsc = 2 * TP / (2 * TP + FP + FN +epsilon)
        specificity = TN / (TN + FP)
        balanced_accuracy = (recall + specificity) / 2
        iou = TP / (TP + FP + FN + epsilon)
        mean_iou = np.sum(iou) / n_masks
        global_iou = np.sum(TP) / np.sum(TP + FP + FN + epsilon)

        bin_class_accuracy = (binTP + binTN) / (binTP + binTN + binFP + binFN + epsilon)
        bin_cat_accuracy = np.sum((binTP + binTN) / (binTP + binTN + binFP + binFN + epsilon))
        bin_global_accuracy = np.sum(binTP + binTN) / (np.sum(binTP + binTN + binFP + binFN) + epsilon)
        bin_weighted_accuracy = np.sum(weights * (binTP + binTN) / (binTP + binTN + binFP + binFN + epsilon))

        bin_precision = binTP / (binTP + binFP + epsilon)
        bin_recall = binTP / (binTP + binFN + epsilon)
        dice_f1 = 2 * binTP / (bin_precision + bin_recall + epsilon)
        dsc = 2 * binTP / (2 * binTP + binFP + binFN + epsilon)
        bin_specificity = binTN / (binTN + binFP)
        balanced_accuracy = (bin_recall + bin_specificity) / 2
        bin_iou = binTP / (binTP + binFP + binFN + epsilon)
        mean_bin_iou = np.sum(bin_iou) / n_masks
        global_bin_iou = np.sum(binTP) / np.sum(binTP + binFP + binFN + epsilon)

        dfs = pd.DataFrame({'Categorical accuracy':[cat_accuracy], 'Global accuracy': [global_accuracy], 'Weighted accuracy' : [weighted_accuracy], 'Mean IOU' : [mean_iou], 'Global IOU' : [global_iou] })
        df = pd.DataFrame({'Class': CLASS_NAMES, 'Class accuracy': class_accuracy, 'Precision': precision, 'Recall' : recall, 'F1 Score (Dice Coefficient)': dsc, 'Specificity':specificity, 'Balanced accuracy':balanced_accuracy, 'IOU': iou})
        return dfs, df






