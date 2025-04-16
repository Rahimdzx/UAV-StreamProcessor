# UAV-StreamProcessor

**UAV-StreamProcessor** is a Python-based system for reading, processing, and visualizing video streams from unmanned aerial vehicles (UAVs). It supports both local files and real-time RTSP streams, and provides a full pipeline from stream ingestion to processed frame delivery over HTTP.

## ✈️ System Overview

The UAV captures video or photo data during flight. This data is streamed or transferred to a central server where it is:

- Saved as a unified MP4 file
- Keyframes are extracted and resized (padded to 640x640 without aspect distortion)
- Selected objects are detected using bounding boxes
- Processed frames are sent via HTTP to a command center endpoint

## ⚙️ Features

- ✅ RTSP or file-based video ingestion  
- ✅ Automatic saving of full video stream  
- ✅ Keyframe extraction and resizing  
- ✅ Object detection with bounding boxes (YOLO, optional)  
- ✅ Frame forwarding via HTTP POST  
- ✅ Optional simulation of RTSP using local files and MediaMTX  

## 🧪 Example Workflow

You can simulate a UAV feed by:
- Using any drone footage (e.g., open-source videos)
- Splitting it into random-length segments
- Simulating an RTSP stream using [MediaMTX](https://github.com/bluenviron/mediamtx) and `ffmpeg`

## 🛠 Dependencies

The system uses the following libraries:

- `numpy`
- `requests`
- `opencv-python` *(for frame processing)*
- `ultralytics` *(optional, for object detection via YOLO)*
- `ffmpeg` *(for video handling)*

Install with:
```bash
pip install numpy requests opencv-python ultralytics
