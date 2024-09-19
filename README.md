My attempt at the Swiss Timing coding challenge

## Solution

This solution supports both the detection and tracking of a person and its segmentation (seperate functionalities).

## Solution 1: Detection and tracking

Uses the MobileNet SSD Caffe pretrained model from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for object detection and the CSRT tracker provided in OpenCV framework for object tracking.
* Confidence considered for object detection model: 0.6
* Libraries used: OpenCV 4.10.0, built from source
* Output stored in a `output/` under same folder as bin

This solution detects and tracks the first object the model classifies as "person", with a confidence above 0.6.

Any other "person" detected is ignored.

### Usage

`AthleteDT tracking <video path> [--visualize]`

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

### Example output 1

**Command:**

`AthleteDT input_video/ice_skating2_4s.mp4`

**Video:**

[output_ice_skating2_4s.webm](https://github.com/PedroM25/AthleteDT/assets/40021588/367b95a8-2f89-4e1f-aaf9-3631b2eeb96d)

**Log file:**

```log
Starting AthleteDT execution.
Successfully imported video "ice_skating2_4s.mp4", FPS: 25.1646, Num frames: 115, Resolution: 1920x1080
FRAME 1: No person detected yet
FRAME 2: No person detected yet
FRAME 3: No person detected yet
FRAME 4: No person detected yet
FRAME 5: No person detected yet
FRAME 6: Person detected, confidence: 0.954797, coordinates: [[1058,337],[1451,337],[1058,922],[1451,922]]
FRAME 6: Target tracked: [[1058,337],[1451,337],[1058,922],[1451,922]]
FRAME 7: Target tracked: [[1049,336],[1450,336],[1049,933],[1450,933]]
FRAME 8: Target tracked: [[1033,320],[1467,320],[1033,966],[1467,966]]
FRAME 9: Target tracked: [[1028,311],[1479,311],[1028,983],[1479,983]]
... snip ...
FRAME 111: Target tracked: [[604,209],[1029,209],[604,842],[1029,842]]
FRAME 112: Target tracked: [[602,219],[1019,219],[602,840],[1019,840]]
FRAME 113: Target tracked: [[605,229],[1014,229],[605,838],[1014,838]]
FRAME 114: Target tracked: [[616,246],[1009,246],[616,831],[1009,831]]
FRAME 115: Target tracked: [[625,254],[1010,254],[625,828],[1010,828]]
No more frames grabbed. Exiting...
Total number of frames processed: 115
Processing time: 21.9272s
```

### Example output 2

**Command:**

`AthleteDT input_video/skate1_4s.mp4`

**Video:**

[output_skate1_4s.webm](https://github.com/PedroM25/AthleteDT/assets/40021588/bb5c66d3-9282-4514-a649-d9480cc4c8b2)


**Log file:**

```log
Starting AthleteDT execution.
Successfully imported video "skate1_4s.mp4", FPS: 25.1751, Num frames: 117, Resolution: 1280x720
FRAME 1: Person detected, confidence: 0.876715, coordinates: [[458,143],[789,143],[458,461],[789,461]]
FRAME 1: Target tracked: [[458,143],[789,143],[458,461],[789,461]]
FRAME 2: Target tracked: [[458,137],[796,137],[458,461],[796,461]]
FRAME 3: Target tracked: [[468,146],[793,146],[468,458],[793,458]]
FRAME 4: Target tracked: [[480,153],[792,153],[480,453],[792,453]]
FRAME 5: Target tracked: [[489,158],[789,158],[489,446],[789,446]]
... snip ...
FRAME 113: Target tracked: [[433,221],[684,221],[433,462],[684,462]]
FRAME 114: Target tracked: [[426,226],[677,226],[426,467],[677,467]]
FRAME 115: Target tracked: [[408,240],[659,240],[408,481],[659,481]]
FRAME 116: Target tracked: [[403,258],[654,258],[403,499],[654,499]]
FRAME 117: Target tracked: [[404,271],[655,271],[404,512],[655,512]]
No more frames grabbed. Exiting...
Total number of frames processed: 117
Processing time: 11.554s
```

## Solution 2: Instance segmentation

Uuses the Mask-RCNN pre-trained model from TensorFlow for object detection. Downloaded from [TensorFlow](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) and the repo for the implementation of Mask-RCNN can be seen [here](https://github.com/matterport/Mask_RCNN)
* Confidence considered for object detection model: 0.8
* Libraries used: OpenCV 4.10.0, built from source
* Output stored in a `output/` under same folder as bin

This solution performs object detection in every frame. Only the first object classified by the model as a "person", with a confidence above 0.8 is considered.

Any other "person" detected is ignored.

## Usage

`AthleteDT segmentation <video path> [--visualize]`

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

### Example output 1

**Command:**

`AthleteDT-seg input_video/ice_skating2_4s.mp4`

**Video:**

[output_ice_skating2_4s.webm](https://github.com/PedroM25/AthleteDT-segmentation/assets/40021588/96b36d1d-8ce9-4224-9dba-40e1cb2560b3)

**Log file:**

```log
Starting AthleteDT-seg execution.
Successfully imported video "ice_skating2_4s.mp4", FPS: 25.1646, Num frames: 115, Resolution: 1920x1080
FRAME 1: Person detected, confidence: 0.995384, coordinates: [[1119,371],[1520,371],[1119,946],[1520,946]]
FRAME 2: Person detected, confidence: 0.988087, coordinates: [[1117,360],[1515,360],[1117,925],[1515,925]]
FRAME 3: Person detected, confidence: 0.996128, coordinates: [[1114,361],[1518,361],[1114,935],[1518,935]]
FRAME 4: Person detected, confidence: 0.998368, coordinates: [[1071,352],[1495,352],[1071,927],[1495,927]]
FRAME 5: Person detected, confidence: 0.999105, coordinates: [[1065,333],[1479,333],[1065,933],[1479,933]]
FRAME 6: Person detected, confidence: 0.999454, coordinates: [[1052,329],[1449,329],[1052,959],[1449,959]]
FRAME 7: Person detected, confidence: 0.999276, coordinates: [[1033,324],[1422,324],[1033,928],[1422,928]]
... snip ...
FRAME 111: Person detected, confidence: 0.996296, coordinates: [[577,225],[948,225],[577,731],[948,731]]
FRAME 112: Person detected, confidence: 0.997915, coordinates: [[581,223],[940,223],[581,728],[940,728]]
FRAME 113: Person detected, confidence: 0.996441, coordinates: [[603,191],[934,191],[603,731],[934,731]]
FRAME 114: Person detected, confidence: 0.99737, coordinates: [[607,214],[935,214],[607,739],[935,739]]
FRAME 115: Person detected, confidence: 0.993023, coordinates: [[614,217],[942,217],[614,745],[942,745]]
No more frames grabbed. Exiting...
Total number of frames processed: 115
Processing time: 246.602s
```

### Example output 2

**Command:**

`AthleteDT-seg input_video/skate1_4s.mp4`

**Video:**

[output_skate1_4s.webm](https://github.com/PedroM25/AthleteDT-segmentation/assets/40021588/94aef04e-30cf-4463-a8cb-fb05b8a2c393)

**Log file:**

```log
Starting AthleteDT-seg execution.
Successfully imported video "skate1_4s.mp4", FPS: 25.1751, Num frames: 117, Resolution: 1280x720
FRAME 1: Person detected, confidence: 0.997617, coordinates: [[469,150],[765,150],[469,426],[765,426]]
FRAME 2: Person detected, confidence: 0.998832, coordinates: [[481,152],[778,152],[481,430],[778,430]]
FRAME 3: Person detected, confidence: 0.99859, coordinates: [[486,164],[775,164],[486,425],[775,425]]
FRAME 4: Person detected, confidence: 0.998743, coordinates: [[492,162],[763,162],[492,435],[763,435]]
FRAME 5: Person detected, confidence: 0.998116, coordinates: [[515,175],[764,175],[515,453],[764,453]]
... snip ...
FRAME 114: Person detected, confidence: 0.992107, coordinates: [[345,231],[674,231],[345,480],[674,480]]
FRAME 115: Person detected, confidence: 0.992937, coordinates: [[313,238],[686,238],[313,497],[686,497]]
FRAME 116: Person detected, confidence: 0.994564, coordinates: [[324,249],[686,249],[324,512],[686,512]]
FRAME 117: Person detected, confidence: 0.974667, coordinates: [[309,253],[692,253],[309,525],[692,525]]
No more frames grabbed. Exiting...
Total number of frames processed: 117
Processing time: 187.847s
```

