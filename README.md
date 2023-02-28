# Python package for Deep Learning and its application in Structural Health Monitoring (DLSHM)

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

The collection contains artificial viaduct images in 1920x1024 <pre>png</pre> images

W Python 

```Python
for student in students:
    print( student)
```

a JavaScript
```JavaScript
for( let student of students) {
 console.log( student);
}
```
