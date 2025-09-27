# Deepfake Video Detection 

This project explores deepfake detection in videos using deep learning.


## Dataset
* Source: [FaceForensices++](https://www.kaggle.com/datasets/hungle3401/faceforensics)
* 200 Fake and 200 Real video cips


## Frame-based Detection
Frame-based classification is carried out, where videos frames are extracted and classified as *real* or *fake*. Video-level labels are then obtained by aggregating frame predictions.

### 1. Frame Extraction
* Extracting a fixed number of frames per video
* Splitting frames into train/validation/test sets

### 2. Baseline CNN
* Implementing a simple CNN trained from scratch on frames
* Model serves as the **baseline** for comparison
* Accuracy = **50%**

### 3. Transfer Learning
* Using MobileNetV2 pretrained on ImageNet as base model
* Fine-tuning last few layers
* Adding data augmentation, dropout, L2 regularization, and learning rate schedules
* Accuracy = ****
