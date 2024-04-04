My attempt at the Swiss Timing coding challenge

## Solution

This solution uses the MobileNet SSD Caffe pretrained model from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for object detection and the CSRT tracker provided in OpenCV framework for object tracking.
* Confidence considered for object detection model: 0.6
* Libraries used: OpenCV 4.9.0, built from source
* Output stored in a `output/` under same folder as bin

This solution detects and tracks the first object the model classifies as "person", with a confidence above 0.6.

Any other "person" detected is ignored.

## Usage

AthleteDT <video path> [--visualize]

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

## Example output

**Command:**

`AthleteDT ../input_video/ice_skating2_4s.mp4`

**Video:**

[ice_skating2_4s.mp4](https://github.com/PedroM25/AthleteDT/assets/40021588/bd3fd4fb-c534-47c4-aef1-e83626e833d1)

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
FRAME 10: Target tracked: [[1023,336],[1474,336],[1023,1008],[1474,1008]]
FRAME 11: Target tracked: [[1008,324],[1487,324],[1008,1037],[1487,1037]]
FRAME 12: Target tracked: [[1013,350],[1473,350],[1013,1035],[1473,1035]]
... snip ...
FRAME 112: Target tracked: [[602,219],[1019,219],[602,840],[1019,840]]
FRAME 113: Target tracked: [[605,229],[1014,229],[605,838],[1014,838]]
FRAME 114: Target tracked: [[616,246],[1009,246],[616,831],[1009,831]]
FRAME 115: Target tracked: [[625,254],[1010,254],[625,828],[1010,828]]
No more frames grabbed. Exiting...
Total number of frames processed: 115
Processing time: 22.0788s

```
