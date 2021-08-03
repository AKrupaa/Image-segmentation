from PIL import Image
import numpy as np
from glob import glob
from unet_model import unet_model
from utils import plot_imgs
from os.path import join, exists, basename
from os import makedirs
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf

# path_to_data = 'TestImages'
path_to_data = 'INPUT_FILES'
path_to_results = 'RESULT_FILES'

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

    return images, labels, label_files


def test_model(display_plot=True, save_predictions=False):
    imgs_np, masks_np, label_files = load_data(path_to_data + "/img", path_to_data + "/mask", image_size)

    x = np.asarray(imgs_np, dtype=np.float32)  # /255
    y = np.asarray(masks_np, dtype=np.float32) / np.max(masks_np)

    y_val = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    x_val = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    input_shape = x_val[0].shape

    model = unet_model(
        input_shape,
        num_classes=1,
        filters=64,
        dropout=0.2,
        num_layers=4,
        output_activation='sigmoid'
    )

    model_filename = 'cp-26.h5'
    model.load_weights(model_filename)

    from time import time

    start = time()
    y_pred = model.predict(x_val)
    end = time()
    print(f'Prediction of {len(y_pred):d} images in {end - start:.2f}s')

    if display_plot:
        p = np.random.permutation(len(x_val))
        num_tests = 8
        fig, _ = plot_imgs(org_imgs=x_val[p], mask_imgs=y_val[p], pred_imgs=y_pred[p], nm_img_to_plot=num_tests)
        # fig.set_size_inches(10, 10)

    if save_predictions:
        if not exists(path_to_results):
            makedirs(path_to_results)

        for i, segmentation in enumerate(y_pred):
            filename = basename(label_files[i])
            filename = filename.replace('roi', 'seg')
            im = Image.fromarray(np.squeeze(segmentation * 255).astype('uint8'), 'L')
            im.save(join(path_to_results, filename))


if __name__ == '__main__':
    # try:
    # Specify an invalid GPU device
    # with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
    # with tf.device('/CPU:0'):
    # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # c = tf.matmul(a, b)
    # history = train_model()
    # for x in range(20):
    test_model(display_plot=True, save_predictions=True)
# except RuntimeError as e:
# print(e)
