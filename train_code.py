import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from unet_model import unet_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from metrics import iou, iou_thresholded
from skimage import color
from pandas import DataFrame
import tensorflow as tf

path_to_data = 'INPUT_FILES'
image_size = (256, 256)


def load_image(fn, img_size):
    im_array = np.array(Image.open(fn).resize(img_size, Image.NEAREST))
    return im_array


def load_data(path_to_raw, path_to_labels, img_size):
    # Load images
    image_files = sorted(glob(path_to_raw + '/*.jpg'))
    images = np.stack([color.rgb2gray(load_image(filename, img_size)) for filename in image_files])

    # Load labels
    label_files = sorted(glob(path_to_labels + '/*.jpg'))
    labels = np.stack([color.rgb2gray(load_image(filename, img_size)) for filename in label_files])

    return images, labels


def train_model():
    imgs_np, masks_np = load_data(path_to_data+"/img", path_to_data+"/mask", image_size)
    # print(np.max(masks_np[12]))
    x = np.asarray(imgs_np, dtype=np.float32) / np.max(imgs_np)
    y = np.round(np.asarray(masks_np, dtype=np.float32) / np.max(masks_np))
    # print(np.max(x))
    # print(np.max(y))
    # y = y.reshape(*y.shape, 1)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

    # Initialize network

    input_shape = x_train[0].shape

    model = unet_model(
        input_shape,
        num_classes=1,
        filters=64,
        dropout=0.2,
        num_layers=4,
        output_activation='sigmoid'
    )

    # Compile + train

    # model_filename = 'unet_model3.h5'
    filepath = "new-cp-{epoch:02d}.h5"
    callback_checkpoint = ModelCheckpoint(
        filepath,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.99),
        loss='binary_crossentropy',
        metrics=[iou, iou_thresholded]
    )

    history = model.fit(
        x_train, y_train, batch_size=6,
        validation_data=(x_val, y_val),
        epochs=30,
        callbacks=[callback_checkpoint, early_stopping],
        shuffle=True
    )

    return history.history


if __name__ == '__main__':

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only use the first GPU
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    # tf.debugging.set_log_device_placement(True)

    # try:
        # Specify an invalid GPU device
        # with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        # with tf.device('/CPU:0'):
            # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            # c = tf.matmul(a, b)
    history = train_model()
    # except RuntimeError as e:
        # print(e)


