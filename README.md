My attempt at the Swiss Timing coding challenge

## Summary
This solution uses the OpenCV library 4.9.0, the MobileNet SSD Caffe pretrained model from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for object detection and the CSRT tracker provided in OpenCV framework for object tracking.
In this case, we are detecting an athlete and tracking them after the initial detection.

In my search I discovered several ways to perform object detection:
* Histogram of gradients (HOG)
* Haar Cascades
* Deep Learning-based object detection using SSD, YOLO, Faster R-CNN

For object tracking there are also several algorithms provided by OpenCV out-of-the-box:
* CSRT
* DaSiamRPN
* GOTURN
* KCF
* MIL
* Nano
* Vit

Source: https://docs.opencv.org/4.9.0/d0/d0a/classcv_1_1Tracker.html

For OpenCV I compiled from source as the libopencv-dev Ubuntu package was outdated (4.5.5 version).

### Object detection

Due to time constraints, I decided to keep it simple and apply Deep Learning-based object detection using a pretrained model.

I used the Caffe implementation of MobileNet SSD. 

Tried to use .pb models from TensorFlow, available in [Kaggle](https://www.kaggle.com/models?task=16686&publisher=tensorflow) but I was unnsuccesful. Adding to this, I also had a hard time finding scaling factors and mean subtraction values associated with models.

I read a few things about having to "freeze" models, found a [page](https://docs.opencv.org/4.x/d1/d8f/tf_cls_tutorial_dnn_conversion.html) explaining how to achieve it but due to time constraints, I decided to just use the most accessible one.

### Object detection - alternatives
I also implemented a HOG-based solution but found it to be too unreliable for person detection. Too many false-positives.

### Object tracking
I used the CSRT tracking algorithm because it was the most reliable and easiest to apply, out of the box, without needing additional parameters.
I tried to use DaSiamRPN but realized I needed to pass in some parameters and once again, due to time constraints, I just used this one.


## Features requested
The following are the features requested and the progress of implementation of each:

|Feature|Progress|
|---|---|
|Choose a sports clip (3-10 seconds)|Tried solution with three different sports clips depicting: fencing, skiing, figure skating|
|Load and process all images|Solution processes video, frame by frame|
|Suitable representation of the tracking results shall be visible in the output |Bounding box|
|Further information can be extracted from the video and displayed (e.g. segmentation, skeleton recognition, ...)|Not performed|
|Information should be available in text format|.log file created after each run|
|Visualized output should be available for playback after processing|.avi file with H.264 codec exported after each run|
|Display output during processing|Yes|


## Future work
Since the clips are short, this solution works fine as it is but tracking of objects for long periods of time can lead to loss of tracking of the original object. To mitigate this, I would perform ocasional object detection and ensure that the program keeps tracking the desired object (the athlete).

I would also explore other pretrained models and dedicate some time to understand what is out there and the different approaches (SSD, Faster R-CNN, YOLO and others)
