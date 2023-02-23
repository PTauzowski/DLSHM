import os.path
import glob
import cv2 as cv
from matplotlib import image as mpimg
import numpy as np

from skimage.color import rgb2lab, lab2rgb

def rgb2labnorm(imgrgb):
    return (rgb2lab(imgrgb) + [0, 128, 128]) / [100, 255, 255]


def labnorm2rgb(chr, col):
    imglab = np.zeros((chr.shape[0], chr.shape[1], 3), dtype=np.float32)
    imglab[:, :, 0] = chr[:, :, 0] * 100
    imglab[:, :, 1:3] = col * 255 - 127
    imgrgb = lab2rgb(imglab)
    return imgrgb


def DirFilesProcessing(source_path, process_callable):
    if source_path == "":
        print("Source path can't be empty")
        return False
    if not os.path.isdir(source_path):
        print("Source path :", source_path + "Is not valid directory name. Processing aborted.")
        return False
    print('Reading images from ', source_path)
    files = glob.glob(source_path + '\\*.*')
    N = len(files)
    if N == 0:
        print("No files found in directory :" + source_path + ". No processing were performed.")
        return False
    print('Number of images N=', N)
    for i, filename in enumerate(files):
        try:
            process_callable(filename)
        except Exception as e:
            print('Cant process image ' + filename + ' because : ' + str(e))
        if i % 100 == 0:
            print('iter=', i, '/', N, flush=True)
    return True


class ImageResizer:

    def __init__(self, sx, sy, destination_path):
        self.destination_path = destination_path

    def __call__(self, filename):
        outfilename = os.path.join(self.destination_path, os.path.basename(filename))
        img = cv.imread(filename, cv.IMREAD_UNCHANGED)
        resized_img = cv.resize(img, (self.destSizeX, self.destSizeY), interpolation=cv.INTER_CUBIC)
        cv.imwrite(outfilename, resized_img)


class ImageResizeDegradation:

    def __init__(self, src, destination_path):
        self.destination_path = destination_path

    def __call__(self, filename):
        outfilename = os.path.join(self.destination_path, os.path.basename(filename))
        img = cv.imread(filename, cv.IMREAD_UNCHANGED)
        halfsized_img = cv.resize(img, ( img.shape[0]//4, img.shape[1]//4), interpolation=cv.INTER_LANCZOS4)
        degraded_img = cv.resize(halfsized_img, (img.shape[0], img.shape[1]), interpolation=cv.INTER_LANCZOS4)
        cv.imwrite(outfilename, degraded_img)


class ImageResizeDegradationPyr:

    def __init__(self, destination_path):
        self.destination_path = destination_path

    def __call__(self, filename):
        outfilename = os.path.join(self.destination_path, os.path.basename(filename))
        img = cv.imread(filename, cv.IMREAD_UNCHANGED)
        img1 = cv.pyrDown(img)
        img2 = cv.pyrDown(img1)
        img3 = cv.pyrUp(img2)
        degraded_img = cv.pyrUp(img3)
        cv.imwrite(outfilename, degraded_img)


class ImageTransformToNumpy:
    def __init__(self, destination_path, imageTransformer):
        self.imageTransformer = imageTransformer
        self.destination_path = destination_path
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    def __call__(self, filename):
        outfilename = os.path.join(self.destination_path, os.path.basename(filename))+'.npy'
        if not os.path.exists(outfilename):
            data_x, data_y = self.imageTransformer(filename)
            with open(outfilename, 'wb') as f:
                np.save(f, data_x)
                np.save(f, data_y)





