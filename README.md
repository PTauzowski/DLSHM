# Deep Learning and its application in Structural Health Monitoring (DLSHM)

The purpose of the package is to perform semantic segmentation of viaduct photos in order to mark structural parts

The following structural parts are detected:
   + 1 - Nonbridge
   + 2 - Slab
   + 3 - Beam
   + 4 - Column
   + 5 - Nonstructural components (Poles, Cables, Fences)
   + 6 - Rail
   + 7 - Sleeper
   + 8 - Other components

In addition, a second semantic segmentation task is also carried out in which concrete damage is detected. The following types of damage are considered:

+  1 - No Damage
+  2 - Concrete Damage (cracks, spalling)
+  3 - Exposed Rebar

Data for semantic segmentation is a set created by:

<i>Narazaki, Y., Hoskere, V., Yoshida, K., Spencer, B. F., & Fujino, Y. (2021). Synthetic environments for vision-based structural condition assessment of Japanese high-speed railway viaducts. Mechanical Systems and Signal Processing, 160, 107850.</i>

The collection contains artificial viaduct dataset images in <code>1920x1024 png</code> images, while masks are stored in <code>640x360 bmp</code>, where numper stored in particular pixel reflects type of structire according to abovementioned lists. There are also depth map stored in image <code>640x360 bmp</code>. Several task has been implemented which differs from input and output size tensor:

+ <code>ICSHM_DEPTH.py</code> - depth estimation with x[640,320,4] (rgb + depth) -> y[640,320,1] depth estimation model
+ <code>ICSHM_DMG.py</code> - segmentation tast with x[640,320,3] (rgb + depth) -> y[640,320,3] concrete damage damage segmentation
+ <code>ICSHM_DMGC.py</code> - segmentation tast with x[640,320,3] (rgb + depth) -> y[640,320,2] concrete damage damage segmentation (cracks only)
+ <code>ICSHM_RGB.py</code> - segmentation tast with x[640,320,3] (rgb ) -> y[640,320,8] structural part segmentation
+ <code>ICSHM_RGBD.py</code> - segmentation tast with x[640,320,4] (rgb + depth) -> y[640,320,8] structural part segmentation with depth channel 
+ <code>ICSHM_RGBHybrid.py</code> -hybrid model -  first depth estimation model is predicted and result together with rgb is input for RGBD prediction.

To perform the above-mentioned tasks, a deep neural network in the U-net architecture was used. Implementation taken from Keras package, the source one can find here: [keras-models](https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/)

Four layers was used in the U-net model so resolution should be divided by 16, therefier images was stretched to resoludion 640x320. Below commen elements of each file was briefly described:

```Python
info_file = pd.read_csv(data_info_file, header=None, index_col=None, delimiter=',')
```
At first the <code>files_train.csv</code> have to be analysed to find out which files will be used in particular task. Different set is used for structural parts segmentation different for damage recognition, but this two sets have non empty intersection. To avoid duplicates <code>files_train.csv</code> contains iformation about proper composition of training set for particular task.

```Python
imgRGB_conv = ICSHM_RGB_Converter(resX, resY)
data_manager = ICSHMDataManager(images_source_path)
data_manager.convert_data_to_numpy_format(imgRGB_conv, train_pathRGB)
```
As the art of preparing data for the tensorflow package teaches us, it is worth storing data for learning not in graphic files, but as matrices of the NumPy package. For this purpose, an object of the <code>Converter</code> class is created which defines how single image have to be transformed. Then <code>DataManager</code> class object is defined which is resposible for all data images manipulation
