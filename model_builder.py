import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Bidirectional, BatchNormalization, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
import util

PRETRAINED_MODELS = {
    'MobileNetV2': (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input, (224,224)),
    'DenseNet121': (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input, (224,224)),
    'EfficientNetB0': (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input, (224,224)),
    'EfficientNetB3': (keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input, (300,300)),
    'ResNet50': (keras.applications.ResNet50, keras.applications.resnet.preprocess_input, (224,224)),
    'InceptionV3': (keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input, (299,299)),
    'Xception': (keras.applications.Xception, keras.applications.xception.preprocess_input, (299,299)),
}

LABELS = ['real','fake']

def get_temporal_model(num_frames, embedding_dim, num_gru=2, gru_units=256, bidirectional=False, num_dense=0, smallest_dense_units=128, batchnorm=False, dropout_rate=0.5):
    '''
    1. All GRUs will have the same number of units.
    2. `smallest_dense_units` is the number of units in the last Dense layer (if any) before output layer
        As more Dense layers are added double the smallest number for each above 
        (e.g. 2 Dense layers - units1 = 256, units2 = 128
              3 Dense layers - units1 = 512, units2 = 256, units3 = 128)
    3. If `batchnorm` is True add BatchNormalization between every pair of GRUs
    4. Dropout is added after GRU, between every pair of Dense layers, and after Dense layer
    '''
    layers = [Input(shape=(num_frames, embedding_dim))]
    # adding GRUs
    for _ in range(num_gru-1):
        gru = GRU(gru_units, return_sequences=True)
        if bidirectional:
            layers.append(Bidirectional(gru))
        else:
            layers.append(gru)
        if batchnorm:
            layers.append(BatchNormalization())
    last_gru = GRU(gru_units, return_sequences=False)
    if bidirectional:
        layers.append(Bidirectional(last_gru))
    else:
        layers.append(last_gru)
    
    # adding Dense & Dropout layers
    add_idx = len(layers)
    dense_units = smallest_dense_units
    for _ in range(num_dense):
        layers.insert(add_idx, Dense(dense_units, activation='relu'))
        if dropout_rate>0.0:
            layers.insert(add_idx, Dropout(dropout_rate))
        dense_units *= 2
    if dropout_rate>0.0:
        layers.append(Dropout(dropout_rate))

    return Sequential(layers)

def get_embeddings(img_model, img_preprocessor, X_frames, num_data, num_frames, img_size):
    base_model = img_model(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(*img_size, 3)
    )
    preprocessed = img_preprocessor(X_frames)
    embeddings = base_model.predict(preprocessed, verbose=1)
    print("Embeddings shape:",embeddings.shape)
    return embeddings.reshape(num_data, num_frames, -1)

def build_classifier(temporal_model, optimizer_name='adam', optimizer_lr=1e-5, momentum=None, nesterov=False):
    
    classifier = Sequential([
        temporal_model,
        Dense(1, activation='sigmoid')
    ])
    if optimizer_name=='adam':
        optimizer = Adam(learning_rate=optimizer_lr)
    elif optimizer_name=='rmsprop':
        optimizer = RMSprop(learning_rate=optimizer_lr)
    elif optimizer_name=='sgd':
        if momentum and nesterov:
            optimizer = SGD(learning_rate=optimizer_lr,
                            momentum=momentum,
                            nesterov=nesterov)
        elif momentum and not nesterov:
            optimizer = SGD(learning_rate=optimizer_lr,
                            momentum=momentum)
        else:
            optimizer = SGD(learning_rate=optimizer_lr)
            
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier
    
def train_test_classifier(
        data_dir, num_frames=5, split_ratio=(0.7, 0.15, 0.15), 
        img_model_name='MobileNetV2', num_gru=2, gru_units=256, 
        bidirectional=False, num_dense=0, smallest_dense_units=128, 
        batchnorm=False, dropout_rate=0.5, epochs=500, batch_size=32, 
        optimizer_name='adam', optimizer_lr=1e-5, momentum=None, nesterov=False,
        estop=EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-5, 
                            patience=10, restore_best_weights=True, verbose=1),
        reduce_lr_on_plateau=None):
    
    img_model, img_preprocessor, img_size = PRETRAINED_MODELS[img_model_name]
    print(f"Image Model: {img_model_name}, Image Size: {img_size}")

    labels, split_3d_data = util.prepare_3d_video_data(data_dir=data_dir,img_size=img_size, num_frames=num_frames,split_ratio=split_ratio)

    print("Video data extraction and splitting completed.")
    print(f"""
Shapes:
X_train: {split_3d_data['X_train'].shape}
X_val: {split_3d_data['X_val'].shape}
X_test: {split_3d_data['X_test'].shape}
y_train: {split_3d_data['y_train'].shape}
y_val: {split_3d_data['y_val'].shape}
X_test: {split_3d_data['y_test'].shape}""")
    
    num_train = split_3d_data['X_train'].shape[0]
    num_val = split_3d_data['X_val'].shape[0]
    num_test = split_3d_data['X_test'].shape[0]
    
    (X_train_frames,_), (X_val_frames,_), (X_test_frames,_) = util.convert_3d_to_2d(split=3,
                                                                                    train=(split_3d_data['X_train'],split_3d_data['y_train']),
                                                                                    val=(split_3d_data['X_val'],split_3d_data['y_val']),
                                                                                    test=(split_3d_data['X_test'],split_3d_data['y_test']))
    
    train_embeddings = get_embeddings(img_model, img_preprocessor, X_train_frames,num_train, num_frames, img_size)
    val_embeddings = get_embeddings(img_model, img_preprocessor, X_val_frames,num_val, num_frames, img_size)
    test_embeddings = get_embeddings(img_model, img_preprocessor, X_test_frames,num_test, num_frames, img_size)

    embedding_dim = train_embeddings.shape[-1]

    temporal_model = get_temporal_model(num_frames, embedding_dim, num_gru, gru_units, bidirectional, num_dense, 
                                        smallest_dense_units, batchnorm, dropout_rate)
    
    temporal_model.summary()
    print("Temporal model defined.")

    classifier = build_classifier(temporal_model, 
                                  optimizer_name, optimizer_lr, 
                                  momentum=None, nesterov=False)
    
    classifier.summary()
    
    callbacks=[]
    if estop and reduce_lr_on_plateau:
        callbacks = [estop, reduce_lr_on_plateau]
    elif estop and not reduce_lr_on_plateau:
        callbacks = [estop]
    
    if callbacks:
        classifier.fit(train_embeddings, split_3d_data['y_train'],
                        validation_data=(val_embeddings, split_3d_data['y_val']),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=1)
    else:
        classifier.fit(train_embeddings, split_3d_data['y_train'],
                        validation_data=(val_embeddings, split_3d_data['y_val']),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    
    return labels, classifier, classifier.evaluate(test_embeddings, split_3d_data['y_test'], verbose=0)


# defaults are given based best results obtained from training
def classifier(video_path, trained_classifier_path, num_frames=16, img_model_name='MobileNetV2'):
    img_model, img_preprocessor, img_size = PRETRAINED_MODELS[img_model_name]
    video_data_3d = util.extract_frames(video_path=video_path, img_size=img_size, num_frames=num_frames)
    print("Shape of 3d video data:", video_data_3d.shape)
    
    video_embeddings = get_embeddings(img_model, img_preprocessor, video_data_3d, 1, num_frames, img_size)
    
    trained_classifier = keras.models.load_model(trained_classifier_path)
    
    prediction = (trained_classifier.predict(video_embeddings)>=0.5).astype(int)[0][0]
    
    # print(prediction)
    
    return LABELS[prediction].upper()

