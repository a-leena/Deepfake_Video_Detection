import joblib
from keras.utils import to_categorical
import numpy as np

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

