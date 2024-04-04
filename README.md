My attempt at the Swiss Timing coding challenge

## Solution

This solution uses the MobileNet SSD Caffe pretrained model from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for object detection and the CSRT tracker provided in OpenCV framework for object tracking.
* Confidence considered for object detection model: 0.6
* Libraries used: OpenCV 4.9.0, built from source
* Output stored in a `output/` under same folder as bin

This solution detects and tracks the first object the model classifies as "person", with a confidence above 0.6.

Any other "person" detected is ignored.

## Usage

`AthleteDT <video path> [--visualize]`

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

## Example output 1

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

## Example output 2

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
