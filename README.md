# Real-Time Object Detection with YOLOv10

## Description
This project implements a real-time object detection system using the YOLOv10 model. It captures video from your webcam, processes each frame to detect objects, and displays the results with bounding boxes and labels. The system runs on CPU and is optimized to maintain a target frame rate for smooth performance.

The model can detect 80 different classes of objects, including people, vehicles, animals, and everyday items. A full list of detectable objects is provided below.

## Detectable Objects (80 classes)

| ID | Class name     |   | ID | Class name   |
| -- | -------------- | - | -- | ------------ |
| 0  | person         |   | 40 | wine glass   |
| 1  | bicycle        |   | 41 | cup          |
| 2  | car            |   | 42 | fork         |
| 3  | motorcycle     |   | 43 | knife        |
| 4  | airplane       |   | 44 | spoon        |
| 5  | bus            |   | 45 | bowl         |
| 6  | train          |   | 46 | banana       |
| 7  | truck          |   | 47 | apple        |
| 8  | boat           |   | 48 | sandwich     |
| 9  | traffic light  |   | 49 | orange       |
| 10 | fire hydrant   |   | 50 | broccoli     |
| 11 | stop sign      |   | 51 | carrot       |
| 12 | parking meter  |   | 52 | hot-dog      |
| 13 | bench          |   | 53 | pizza        |
| 14 | bird           |   | 54 | donut        |
| 15 | cat            |   | 55 | cake         |
| 16 | dog            |   | 56 | chair        |
| 17 | horse          |   | 57 | couch        |
| 18 | sheep          |   | 58 | potted plant |
| 19 | cow            |   | 59 | bed          |
| 20 | elephant       |   | 60 | dining table |
| 21 | bear           |   | 61 | toilet       |
| 22 | zebra          |   | 62 | tv           |
| 23 | giraffe        |   | 63 | laptop       |
| 24 | backpack       |   | 64 | mouse        |
| 25 | umbrella       |   | 65 | remote       |
| 26 | handbag        |   | 66 | keyboard     |
| 27 | tie            |   | 67 | cell phone   |
| 28 | suitcase       |   | 68 | microwave    |
| 29 | frisbee        |   | 69 | oven         |
| 30 | skis           |   | 70 | toaster      |
| 31 | snowboard      |   | 71 | sink         |
| 32 | sports ball    |   | 72 | refrigerator |
| 33 | kite           |   | 73 | book         |
| 34 | baseball bat   |   | 74 | clock        |
| 35 | baseball glove |   | 75 | vase         |
| 36 | skateboard     |   | 76 | scissors     |
| 37 | surfboard      |   | 77 | teddy bear   |
| 38 | tennis racket  |   | 78 | hair drier   |
| 39 | bottle         |   | 79 | toothbrush   |

## Setup

### Requirements
- Python 3.7 or higher
- Webcam connected to your computer

### Install Dependencies
Install the required Python packages using pip:

```bash
pip install opencv-python ultralytics
```

### Model File
Ensure the YOLOv10 model file `yolov10n.pt` is placed in the project root directory.

## Usage

Run the real-time object detection script:

```bash
python rt_object_detection.py
```

- The script will open a window showing the webcam feed with detected objects highlighted.
- Bounding boxes and class labels with confidence scores are displayed on detected objects.
- The current FPS (frames per second) is shown on the video.
- Press the `q` key to quit the application.

## Implementation Details

- **Snapshot Saving:**
  - When objects are detected in a frame, a snapshot image is saved in the `snapshots/` folder with a timestamped filename.

- **Performance:**
  - The script limits the frame processing rate to the target FPS to balance performance and detection accuracy.

## Snapshots Folder

The `snapshots/` directory contains images of frames where objects were detected. These snapshots can be used for further analysis or record-keeping.

---

This project provides a simple and effective way to perform real-time object detection using a lightweight YOLOv10 model on CPU, suitable for various applications such as surveillance, monitoring, and interactive systems.