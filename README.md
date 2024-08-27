# OpenCV Projects in Python

This repository contains several projects that demonstrate the usage of the OpenCV library in Python. The focus is on different aspects of image and video processing, including object detection, face recognition, and video frame manipulation. Below is a detailed description of each file and its functionality within the repository.

## Files and Directories

### 1. `autopilot.py`
This script is designed to detect and highlight road boundaries in a provided video clip using the OpenCV library. The approach is entirely based on traditional computer vision techniques without the use of machine learning or neural networks. The main functionality involves the drawing of enclosing lines that outline the detected road edges, helping to simulate a basic autopilot system.

### 2. `faceID.py`
In this project, a pre-trained [Haar Cascade](https://github.com/opencv/opencv?tab=readme-ov-file#readme) classifier is utilized to detect faces within an image. Once a face is detected, the corresponding frame is edited and then passed through a custom neural network for classification. The neural network, trained separately within the repository, determines whether the detected face belongs to the user or not ("me" vs. "not me"). The neural network and its training code can be found in the `ML` directory.

### 3. `img_processing.py`
This file contains a collection of tools and functions for various image processing tasks using OpenCV. These utilities cover a range of operations such as filtering, transformations, and enhancements, making it a useful resource for working with images in Python.

### 4. `object_recogn_without_cascades.py`
This script focuses on the recognition of a car's license plate from an image using only OpenCV. Unlike other methods that rely on pre-trained models or classifiers, this approach uses traditional image processing techniques to detect and extract the license plate without the assistance of cascades or neural networks.

### 5. `plate_number_recogn.py`
Here, a pre-trained [Haar Cascade](https://github.com/opencv/opencv?tab=readme-ov-file#readme) classifier is employed to detect and recognize license plates in images. The process is automated and allows for efficient recognition of plates by leveraging the pre-trained model, demonstrating a straightforward method for plate number recognition using OpenCV.

### 6. `split_video_frames.py`
This utility is used to split a given video file into individual frames. The frames are extracted and saved in a specified directory, enabling further processing or analysis on a frame-by-frame basis. This script is particularly useful for tasks that require precise manipulation of video content at the frame level.

### 7. `video_prep_from_front_cam.py`
This script provides several video editing functions specifically tailored for footage captured from a front-facing camera. It includes features such as adding a background, blurring the face, and highlighting the face within the video. These capabilities make it useful for privacy protection or enhancing video content where the subject's face is a focal point.

---

Each of these scripts demonstrates different capabilities of the OpenCV library, ranging from basic image processing to more complex tasks like object recognition and video editing. Whether you are a beginner looking to learn OpenCV or an experienced developer seeking practical examples, this repository offers valuable insights and
