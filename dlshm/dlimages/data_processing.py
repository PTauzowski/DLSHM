import os
import numpy as np
import cv2 as cv
from skimage.transform import resize
from PIL import Image
import pandas as pd
from matplotlib import image as mpimg, pyplot as plt

from dlshm.dlimages.convert import rgb2labnorm, labnorm2rgb


class ICSHM_RGB_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 8),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        mask = np.asarray(mpimg.imread(labName))
        mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask == 2, 1, 0)
        self.y[:, :, 2] = np.where(mask == 3, 1, 0)
        self.y[:, :, 3] = np.where(mask == 4, 1, 0)
        self.y[:, :, 4] = np.where(mask == 5, 1, 0)
        self.y[:, :, 5] = np.where(mask == 6, 1, 0)
        self.y[:, :, 6] = np.where(mask == 7, 1, 0)
        self.y[:, :, 7] = np.where(mask == 8, 1, 0)

        return self.x, self.y

class ICSHM_BG_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 2),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        mask = np.asarray(mpimg.imread(labName))
        mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask != 1, 1, 0)

        return self.x, self.y

class ICSHM_RGB_FULL_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 8),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        mask = np.asarray(mpimg.imread(labName))
        #  mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask == 2, 1, 0)
        self.y[:, :, 2] = np.where(mask == 3, 1, 0)
        self.y[:, :, 3] = np.where(mask == 4, 1, 0)
        self.y[:, :, 4] = np.where(mask == 5, 1, 0)
        self.y[:, :, 5] = np.where(mask == 6, 1, 0)
        self.y[:, :, 6] = np.where(mask == 7, 1, 0)
        self.y[:, :, 7] = np.where(mask == 8, 1, 0)

        return self.x, self.y

class ICSHM_RGBD_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 4),dtype=np.float32)
        self.y = np.empty((resY, resX, 8),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        depth_array = resize(mpimg.imread(depthName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 3] = depth_array

        mask = np.asarray(mpimg.imread(labName))
        mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask == 2, 1, 0)
        self.y[:, :, 2] = np.where(mask == 3, 1, 0)
        self.y[:, :, 3] = np.where(mask == 4, 1, 0)
        self.y[:, :, 4] = np.where(mask == 5, 1, 0)
        self.y[:, :, 5] = np.where(mask == 6, 1, 0)
        self.y[:, :, 6] = np.where(mask == 7, 1, 0)
        self.y[:, :, 7] = np.where(mask == 8, 1, 0)

        return self.x, self.y

class ICSHM_Depth_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 1),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        depth_array = resize(mpimg.imread(depthName), (self.resY, self.resX), anti_aliasing=True)
        self.y[:, :, 0] = depth_array

        return self.x, self.y

class ICSHM_DMG_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 3),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        mask = np.asarray(mpimg.imread(dmgName))
        mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask == 2, 1, 0)
        self.y[:, :, 2] = np.where(mask == 3, 1, 0)

        return self.x, self.y

class ICSHM_DMGC_Converter:
    def __init__(self,resX,resY):
        self.resX=resX
        self.resY=resY
        self.x = np.empty((resY, resX, 3),dtype=np.float32)
        self.y = np.empty((resY, resX, 2),dtype=np.float32)

    def __call__(self, imageName, labName, dmgName, depthName):
        image_array = resize(cv.imread(imageName), (self.resY, self.resX), anti_aliasing=True)
        self.x[:, :, 0] = image_array[:, :, 0]
        self.x[:, :, 1] = image_array[:, :, 1]
        self.x[:, :, 2] = image_array[:, :, 2]

        mask = np.asarray(mpimg.imread(dmgName))
        mask = Image.fromarray(mask).resize((self.resX, self.resY), Image.NEAREST)
        mask = np.array(mask)
        self.y[:, :, 0] = np.where(mask == 1, 1, 0)
        self.y[:, :, 1] = np.where(mask == 2, 1, 0)

        return self.x, self.y

class ICSHMDataManager:
    def __init__(self, tokaido_path, csv_ind=5 ):
        self.tokaido_path=tokaido_path
        self.data_csv = pd.read_csv(tokaido_path + r'/files_train.csv', header=None, index_col=None, delimiter=',')
        col_valid = self.data_csv[csv_ind]
        self.idx_valid = [i for i in range(len(col_valid)) if col_valid[i]]
        self.filenames = [self.data_csv.iloc[i][0] for i in range(len(col_valid)) if col_valid[i]]

    def convert_data_to_numpy_format(self, process, dataset_path):
        isExist = os.path.exists(dataset_path)
        if not isExist:
            os.makedirs(dataset_path)
        N=len(self.idx_valid)
        for i, idx in enumerate(self.idx_valid):
            filename = os.path.join(dataset_path,os.path.basename(self.data_csv.iloc[idx][0]))+'.npy'
            if not os.path.exists(filename):
                try:
                    imageName = os.path.join(self.tokaido_path, self.data_csv.iloc[idx][0])
                    labName = os.path.join(self.tokaido_path, self.data_csv.iloc[idx][1])
                    dmgName = os.path.join(self.tokaido_path, self.data_csv.iloc[idx][2])
                    depthName = os.path.join(self.tokaido_path, self.data_csv.iloc[idx][3])

                    x, y = process(imageName, labName, dmgName, depthName )

                    if i==0:
                        self.class_pixel_counts = np.sum(y, axis=(0, 1))
                        self.total_pixels = np.sum(self.class_pixel_counts)
                    else:
                        self.class_pixel_counts = self.class_pixel_counts + np.sum(y, axis=(0, 1))
                        self.total_pixels = self.total_pixels + np.sum(self.class_pixel_counts)

                    with open(filename, 'wb') as f:
                        np.save(f, x)
                        np.save(f, y)
                except Exception as err:
                    print('Cant import ' + imageName + ' because:', err )
                if i % 100 == 0:
                    print('iter=', i, '/', N, flush=True)
            else:
                f=open(filename, 'rb')
                x=np.load(f)
                y=np.load(f)
                if i == 0:
                    self.class_pixel_counts = np.sum(y, axis=(0, 1))
                    self.total_pixels = np.sum(self.class_pixel_counts)
                else:
                    self.class_pixel_counts = self.class_pixel_counts + np.sum(y, axis=(0, 1))
                    self.total_pixels = self.total_pixels + np.sum(self.class_pixel_counts)

        self.weights = self.total_pixels / (self.class_pixel_counts + 1e-6)
        self.weights =  self.weights / sum(self.weights)

    #class_weights = total_pixels / (class_pixel_counts + 1e-8)  # Add a small value to prevent division by zero
    #return class_weights / np.sum(class_weights)

def get_data(self):
        return self.filenames


def compute_class_weights(masks):

    # Count the pixels for each class
    class_pixel_counts = np.sum(masks, axis=(0, 1))

    # Total pixels in the image
    total_pixels = np.sum(class_pixel_counts)

    # Compute weights (inverse frequency)
    weights = total_pixels / (class_pixel_counts + 1e-6)  # Add small value to avoid division by zero

    # Optionally normalize weights to sum to 1
    weights = weights / np.sum(weights)

    return weights




