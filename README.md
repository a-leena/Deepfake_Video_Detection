# Deepfake Video Detection 
This project explores deepfake detection in videos using deep learning.


## Dataset
* Source: [Curated DFD (Deep Fake Detection)](https://www.kaggle.com/datasets/hungle3401/faceforensics)
* 200 Fake and 200 Real video cips


## Frame Extraction
* Extracting a fixed number of frames per video and representing each video as a 3D matrix
* Splitting the data into train/validation/test sets

## Frame-based Detection
Frame-based classification is carried out, where video frames are extracted and classified as *real* or *fake*. Video-level labels are then obtained by simply aggregating frame predictions.

### 1. Baseline CNN
* Implementing a simple CNN trained from scratch on frames
* Model serves as the **baseline** for comparison
* Accuracy = **50%**

### 2. Transfer Learning
* Using MobileNetV2 & XceptionNet, both pretrained on ImageNet, as base models
* Fine-tuning last few layers
* Adding data augmentation and dropout
* Best Accuracy = **62%** (achieved by **MobileNetV2**)

## Video-based Detection
Video-based classification is carried out, where the 3D data is directly fed into the model to classify the video as *real* or *fake*. 

### 3D CNN
* Directly learns the spatio-temporal features
* Training was computationally very expensive
* Accuracy = **50%**
