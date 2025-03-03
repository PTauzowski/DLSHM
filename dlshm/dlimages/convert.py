import os.path
import glob
import cv2 as cv
from matplotlib import image as mpimg
import numpy as np
import tensorflow as tf
from multiprocessing import Pool, Manager, freeze_support
from skimage.color import rgb2lab, lab2rgb
import random

def rgb2labnorm(imgrgb):
    return (rgb2lab(imgrgb) + [0, 128, 128]) / [100, 255, 255]


def labnorm2rgb(chr, col):
    imglab = np.zeros((chr.shape[0], chr.shape[1], 3), dtype=np.float32)
    imglab[:, :, 0] = chr[:, :, 0] * 100
    imglab[:, :, 1:3] = col * 255 - 127
    imgrgb = lab2rgb(imglab)
    return imgrgb

class RandomIntParameter:
    def __init__(self,lb,ub):
        self.lb=lb
        self.ub=ub

    def get(self):
        return random.randint(self.lb, self.ub)

class RandomDoubleParameter:
    def __init__(self,lb,ub):
        self.lb=lb
        self.ub=ub

    def get(self):
        return random.uniform(self.lb, self.ub)

class RandomSetParameter:
    def __init__(self,set):
        self.set=set

    def get(self):
        return self.set[ random.randint(0, len(self.set)-1) ]

def compose_file_name(filename, ext, forceUniqueNames=False):
    outfilename = filename + "." + ext
    basename=filename
    if (forceUniqueNames):
        while os.path.exists(outfilename):
            basename=basename+"_1"
            outfilename = basename + "." + ext
    return outfilename


def write_numpy_image_files(filename, data_x, data_y, preview=False, forceUniqueNames=False):
    if ( preview ):
        outfilename_x = compose_file_name(filename + '_x', 'png', forceUniqueNames)
        outfilename_y = compose_file_name(filename + '_y', 'png', forceUniqueNames)
        cv.imwrite(outfilename_x, data_x * 255)
        cv.imwrite(outfilename_y, data_y * 255)
    else:
        outfilename = compose_file_name(filename, '.npy', forceUniqueNames)
        if not os.path.exists(outfilename):
            with open(outfilename, 'wb') as f:
                np.save(f, data_x)
                np.save(f, data_y)

def dir_files_processing(source_path, process_callable, Nlimit=-1):
    if source_path == "":
        print("Source path can't be empty")
        return False
    if not os.path.exists(source_path):
        print("Source path :", source_path + " not exists. Processing aborted.")
        return False
    if not os.path.isdir(source_path):
        print("Source path :", source_path + "Is not valid directory name. Processing aborted.")
        return False
    print('Reading images from ', source_path)
    files = glob.glob(source_path + '/*')
    if Nlimit != -1 :
        files=files[0:min(Nlimit,len(files))-1]
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

def process_file_in_parallel(filename, process_callable, progress):
    try:
        # Your processing logic here
        # For example, you can call your existing process_callable function
        process_callable(filename)
        with progress.get_lock():
            progress.value += 1
            if progress.value % 100 == 0:
                print(f'Processed {progress.value} out of {N} files')
    except Exception as e:
        print(f"Can't process image {filename} because: {str(e)}")

def dir_files_parallel_processing(source_path, process_callable, Nlimit=-1, num_processes=8):
    if source_path == "":
        print("Source path can't be empty")
        return False
    if not os.path.isdir(source_path):
        print("Source path :", source_path + "Is not valid directory name. Processing aborted.")
        return False
    print('Reading images from ', source_path)
    files = glob.glob(source_path + '\\*.*')
    if Nlimit != -1 :
        files=files[0:min(Nlimit,len(files))-1]
    N = len(files)
    if N == 0:
        print("No files found in directory :" + source_path + ". No processing were performed.")
        return False
    print('Number of images N=', N)
    progress = Manager().Value('i', 0)
    result_queue = Queue()
    for file in files:
        pool.apply_async(process_file_in_parallel, (file, progress, result_queue))

        # Wait for all processes to finish
        pool.close()
        pool.join()

        # Check results and report if some files failed
        failed_files = []
        for _ in range(len(files)):
            result = result_queue.get()
            if result is None:
                failed_files.append(result)

        if failed_files:
            print(f"Processing failed for {len(failed_files)} files.")
    return True


class ImageResizer:

    def __init__(self, destSizeX, destSizeY, destination_path):
        self.destination_path = destination_path
        self.destSizeX=destSizeX
        self.destSizeY=destSizeY

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

class ImageTrainingPairMultiscaleAugmented:
    def __init__(self, resX, resY, scaleFactor):
        self.resX = resX
        self.resY = resY
        self.bigResX = resX * scaleFactor
        self.bigResY = resY * scaleFactor
        self.scaleFactor = scaleFactor
        self.data_x = np.zeros((resY, resX, 3), dtype=np.float32)
        self.data_y = np.zeros((self.bigResY, self.bigResX, 3), dtype=np.float32)
        self.aspect=resX/resY


    def get_images(self, image):
        temp = RandomIntParameter(self.bigResX*2, image.shape[1])
        cResX = temp.get()
        cResY = round(cResX*self.aspect)
        tempX=RandomIntParameter(0,image.shape[1]-cResX)
        tempY=RandomIntParameter(0,image.shape[0]-cResY)
        x0=tempX.get()
        y0=tempY.get()
        self.data_y = cv.resize(image[y0:y0+cResY,x0:x0+cResX, :], (self.bigResY, self.bigResX), cv.IMREAD_UNCHANGED)
        self.data_x = cv.resize(self.data_y, (self.resY, self.resX), cv.IMREAD_UNCHANGED)
        return self.data_x, self.data_y



class ImageTrainingPairSingleScale:
    def __init__(self, resX, resY, scaleFactor):
        self.resX = resX
        self.resY = resY
        self.bigResX = resX * scaleFactor
        self.bigResY = resY * scaleFactor
        self.scaleFactor = scaleFactor
        self.data_x = np.zeros((resY, resX, 3), dtype=np.float32)
        self.data_y = np.zeros((self.bigResY, self.bigResX, 3), dtype=np.float32)
        self.aspect=resX/resY


    def get_images(self, image):
        temp = RandomIntParameter(self.bigResX*2, image.shape[1])
        cResX = temp.get()
        cResY = round(cResX*self.aspect)
        tempX=RandomIntParameter(0,image.shape[1]-cResX)
        tempY=RandomIntParameter(0,image.shape[0]-cResY)
        x0=tempX.get()
        y0=tempY.get()
        self.data_y = cv.resize(image[y0:y0+cResY,x0:x0+cResX, :], (self.bigResY, self.bigResX), cv.IMREAD_UNCHANGED)
        self.data_x = cv.resize(self.data_y, (self.resY, self.resX), cv.IMREAD_UNCHANGED)
        return self.data_x, self.data_y

class SingleImageSaveAsNumpy:
    def __init__(self, destination_path ):
        self.destination_path = destination_path
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    def __call__(self, filename):
        outfilename = os.path.join(self.destination_path, os.path.basename(filename))+'.npy'
        if not os.path.exists(outfilename):
            image = cv.imread(filename, cv.IMREAD_UNCHANGED)/255
            with open(outfilename, 'wb') as f:
                np.save(f, image)

class MosaicImageTransformer:
    def __init__(self, resX, resY, upscaleFactor, divFactor ):
        self.resX=resX
        self.resY=resY
        self.aspect=resX/resY
        self.upscaleFactor = upscaleFactor
        self.divFactor = divFactor

    def __call__(self, filename):
        image = cv.imread(filename, cv.IMREAD_UNCHANGED)/255
        if image.shape[0]>image.shape[1]:
            widthX = image.shape[0]//self.divFactor
            widthY = round(widthX/self.aspect)
        else:
            widthY = image.shape[1]//self.divFactor
            widthX = round(widthY * self.aspect)
        nx = image.shape[0]//widthX
        ny = image.shape[1]//widthY
        nimg = nx*ny
        data_x = np.zeros((nimg,self.resY//self.upscaleFactor, self.resX//self.upscaleFactor, 3), dtype=np.float32)
        data_y = np.zeros((nimg,self.resY, self.resX, 3), dtype=np.float32)
        x0=(image.shape[0] - nx * widthX) // 2
        y0=(image.shape[1] - ny * widthY) // 2
        ind=0
        for i in range(0,nx):
            for j in range(0, ny):
                x = x0 + i * widthX
                y = y0 + j * widthY
                data_y[ind,] = cv.resize(image[y:y+widthX, x:x+widthY,:], (self.resY, self.resX), cv.INTER_LANCZOS4)
                data_x[ind,] = cv.resize(data_y[ind, ], (self.resY // self.upscaleFactor, self.resX // self.upscaleFactor), cv.INTER_LANCZOS4)
                ind=ind+1
        return data_x, data_y

class TrainingPairTransformToNumpy:
    def __init__(self, destination_path, imageTransformer, preview=False, forceUniqueNames=False):
        self.imageTransformer = imageTransformer
        self.destination_path = destination_path
        self.preview=preview
        self.forceUniqueNames = forceUniqueNames
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    def __call__(self, filename):
        outfilename=outfilename = os.path.join(self.destination_path, os.path.basename(filename))
        data_x, data_y = self.imageTransformer(filename)
        write_numpy_image_files(outfilename, data_x, data_y, self.preview)

class TrainingMultiPairTransformToNumpy:
    def __init__(self, destination_path, imageTransformer, preview=False, forceUniqueNames=False):
        self.imageTransformer = imageTransformer
        self.destination_path = destination_path
        self.preview=preview
        self.forceUniqueNames = forceUniqueNames
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    def __call__(self, filename):
        data_x, data_y = self.imageTransformer(filename)
        for k in range(0,data_x.shape[0]):
            outfilename = os.path.join(self.destination_path, os.path.basename(filename) + f"_{k}_")
            write_numpy_image_files(outfilename, data_x[k,], data_y[k,], self.preview, self.forceUniqueNames)



def gener_training_pair_transform_to_numpy(pathName, data_generator, scope=-1):
    if not os.path.exists(pathName):
        os.makedirs(pathName)
    N=scope
    if N==-1:
        N=len(data_generator)
    k=0;
    for data_x, data_y in data_generator:
        for b in range(0,data_x.shape[0]):
            with open(os.path.join(pathName, f"genPair_{k}_{b}.npy"), 'wb') as f:
                np.save(f, data_x)
                np.save(f, data_y)
        k+=1
        if k>N:
            break

def image_components_presentation(imagePath, outputDir):
    image_arrayRGB = cv.imread(imagePath)
    #image_arrayB, image_arrayG, image_arrayR = cv.split(image_arrayRGB)

    lab = cv.cvtColor(image_arrayRGB, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab)

    image_arrayLab = rgb2labnorm(image_arrayRGB/255)
    image_arrayR=image_arrayRGB.copy()
    image_arrayG=image_arrayRGB.copy()
    image_arrayB=image_arrayRGB.copy()
    image_arrayR[:, :, 1] = 0
    image_arrayR[:, :, 2] = 0
    image_arrayG[:, :, 0] = 0
    image_arrayG[:, :, 2] = 0
    image_arrayB[:, :, 0] = 0
    image_arrayB[:, :, 1] = 0

    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_R.png', image_arrayR)
    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_G.png', image_arrayG)
    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_B.png', image_arrayB)

    image_arrayL = lab.copy()
    image_arraya = lab.copy()
    image_arrayb = lab.copy()

    image_arrayL[:, :, 0] = L
    image_arrayL[:, :, 1] = L
    image_arrayL[:, :, 2] = L

    image_arraya[:, :, 0] = 255-A
    image_arraya[:, :, 1] = 255-A
    image_arraya[:, :, 2] = A

    image_arrayb[:, :, 0] = 256-B
    image_arrayb[:, :, 1] = B
    image_arrayb[:, :, 2] = B


    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_L.png', image_arrayL  )
    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_cr.png', image_arraya )
    cv.imwrite(os.path.join(outputDir, os.path.basename(imagePath)) + '_yb.png', image_arrayb )



