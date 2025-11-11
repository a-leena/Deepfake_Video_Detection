import os
import numpy as np
import pandas as pd
import cv2
import joblib
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def extract_frames(video_path, img_size=(224,224), num_frames=5):
    # print(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames//num_frames, 1)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*frame_interval)
        ret, frame = cap.read()
        # break the loop if end of the video is reached
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)
    cap.release()
    # fill any missing frames with blank frames
    while len(frames) < num_frames:
        frames.append(np.zeros(img_size+(3,), np.uint8))
    return np.array(frames)

def train_val_test_video_splits(data_dir, img_size=(224,224), num_frames=5, split_ratio=(0.7, 0.15, 0.15)):
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for cls in ['real', 'fake']:

        source = os.path.join(data_dir, cls)
        all_files = os.listdir(source)
        train_files, val_test_files = train_test_split(all_files, train_size=split_ratio[0], random_state=42)
        val_ratio = split_ratio[1]/(split_ratio[1] + split_ratio[2])
        val_files, test_files = train_test_split(val_test_files, train_size=val_ratio, random_state=42)
        cls_data = {'train':[], 'val':[], 'test':[]}
        for split, files in zip(['train', 'val', 'test'],
                                [train_files, val_files, test_files]):
            print(f"{split.upper()} set: {len(files)} videos")
            for file in files:
                cls_data[split].append(extract_frames(os.path.join(source, file), img_size=img_size, num_frames=num_frames))

        X_train.extend(cls_data['train'])
        X_val.extend(cls_data['val'])
        X_test.extend(cls_data['test'])
        y_train.extend([cls]*len(cls_data['train']))
        y_val.extend([cls]*len(cls_data['val']))
        y_test.extend([cls]*len(cls_data['test']))
    
    split_3d_data = {
        'X_train':np.array(X_train),
        'X_val':np.array(X_val),
        'X_test':np.array(X_test),
        'y_train':np.array(y_train),
        'y_val':np.array(y_val),
        'y_test':np.array(y_test)
    }
    return split_3d_data

def target_factorization(split, y=None, y_train=None, y_val=None, y_test=None):
    '''
    `split` indicates the number of subsets of y
    `y` is the entire target series, provided if `split=1`
    if `split=2` we will have `y_train` and `y_test`
    if `split=3` we will have `y_train`, `y_val`, and `y_test`
    `y_train` is the target series for training data
    `y_val` is the target series for validation data
    `y_test` is the target series for testing data

    Function returns the labels as well as the factorized target data
    '''
    if split==1:
        y, labels = pd.factorize(y)
        return labels, y
    if split==2:
        y_train, labels = pd.factorize(y_train)
        y_test = pd.factorize(y_test)[0]
        return labels, y_train, y_test
    if split==3:
        y_train, labels = pd.factorize(y_train)
        y_val = pd.factorize(y_val)[0]
        y_test = pd.factorize(y_test)[0]
        return labels, y_train, y_val, y_test
    return None

def prepare_3d_video_data(data_dir, img_size=(224,224), num_frames=5, split_ratio=(0.7, 0.15, 0.15)):
    if data_dir==os.path.join('data', 'DFD') and num_frames==16 and img_size==(224,224):
        X_train, X_val, X_test, y_train, y_val, y_test = load_split_3d_data(os.path.join('artifacts', 'split_3d_data.pkl'))
        split_3d_data = {
            'X_train':X_train,
            'X_val':X_val,
            'X_test':X_test,
            'y_train':y_train,
            'y_val':y_val,
            'y_test':y_test
        }
    else:
        split_3d_data = train_val_test_video_splits(data_dir=data_dir, img_size=img_size, num_frames=num_frames, split_ratio=split_ratio)
    labels, split_3d_data['y_train'], split_3d_data['y_val'], split_3d_data['y_test'] = target_factorization(
        split=3, 
        y_train=split_3d_data['y_train'],
        y_val=split_3d_data['y_val'],
        y_test=split_3d_data['y_test']
    )
    return labels, split_3d_data

def convert_3d_to_2d(split, data=None, train=None, val=None, test=None):
    '''
    `split` indicates the number of subsets of data
    if `split=1` then entire data is provided as tuple `(X,y)`
    if `split=2` we will have `(X_train,y_train)` and `(X_test,y_test)`
    if `split=3` we will have `X_train,y_train)`, `(X_val,y_val)`, and `(X_test,y_test)`
    '''
    if split==1:
        X, y = data
        num_frames, img_size, channels = X.shape[1], X.shape[2:4], X.shape[4]
        X_frames = X.reshape(-1,img_size[0], img_size[1], channels)
        y_frames = np.repeat(y, num_frames, axis=0)
        return (X_frames, y_frames)
    if split==2:
        X_train, y_train = train
        X_test, y_test = test
        num_frames, img_size, channels = X_train.shape[1], X_train.shape[2:4], X_train.shape[4]
        X_train_frames = X_train.reshape(-1,img_size[0], img_size[1], channels)
        y_train_frames = np.repeat(y_train, num_frames, axis=0)
        X_test_frames = X_test.reshape(-1,img_size[0], img_size[1], channels)
        y_test_frames = np.repeat(y_test, num_frames, axis=0)
        return (X_train_frames, y_train_frames), (X_test_frames, y_test_frames)
    if split==3:
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test
        num_frames, img_size, channels = X_train.shape[1], X_train.shape[2:4], X_train.shape[4]
        X_train_frames = X_train.reshape(-1,img_size[0], img_size[1], channels)
        y_train_frames = np.repeat(y_train, num_frames, axis=0)
        X_val_frames = X_val.reshape(-1,img_size[0], img_size[1], channels)
        y_val_frames = np.repeat(y_val, num_frames, axis=0)
        X_test_frames = X_test.reshape(-1,img_size[0], img_size[1], channels)
        y_test_frames = np.repeat(y_test, num_frames, axis=0)
        return (X_train_frames, y_train_frames), (X_val_frames, y_val_frames), (X_test_frames, y_test_frames)
    return None

def load_split_3d_data(data_path):
    split_3d_data = joblib.load(data_path)
    X_train = split_3d_data['X_train']
    X_val = split_3d_data['X_val']
    X_test = split_3d_data['X_test']
    y_train = split_3d_data['y_train']
    y_val = split_3d_data['y_val']
    y_test = split_3d_data['y_test']
    return X_train, X_val, X_test, y_train, y_val, y_test

def one_hot_encoding(num_categories, split, y=None, y_train=None, y_val=None, y_test=None):
    '''
    `split` indicates the number of subsets of y
    `y` is the entire target series, provided if `split=1`
    if `split=2` we will have `y_train` and `y_test`
    if `split=3` we will have `y_train`, `y_val`, and `y_test`
    `y_train` is the target series for training data
    `y_val` is the target series for validation data
    `y_test` is the target series for testing data
    '''
    if split==1:
        y = to_categorical(y, num_categories)
        return y
    if split==2:
        y_train = to_categorical(y_train, num_categories)
        y_test = to_categorical(y_test, num_categories)
        return y_train, y_test
    if split==3:
        y_train = to_categorical(y_train, num_categories)
        y_val = to_categorical(y_val, num_categories)
        y_test = to_categorical(y_test, num_categories)
        return y_train, y_val, y_test
    return None
