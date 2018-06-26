import matplotlib
matplotlib.use('Agg')

import os.path as osp
import os
from tensorflow.examples.tutorials.mnist import input_data
import gzip
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imsave
import matplotlib.pyplot as plt


DATA_PATH = osp.join('common', 'data')
MNIST_DATA_PATH = osp.join(DATA_PATH, 'mnist')

def load_mnist(dataset_name='mnist'):
    data_dir = osp.join(DATA_PATH, dataset_name)
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(osp.join(data_dir, 'train-images-idx3-ubyte.gz'), 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(osp.join(data_dir, 'train-labels-idx1-ubyte.gz'), 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(osp.join(data_dir, 't10k-images-idx3-ubyte.gz'), 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(osp.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 818
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def load_ocr(dataset_name='ocr_eng_vertical_1000', input_shape=(120, 16, 3), validate_dataset_name=None, sample_num=4):
    data_dir = osp.join(DATA_PATH, dataset_name)
    image_names = os.listdir(osp.join(data_dir, 'images'))
    images = []
    masks = []
    for each_img_name in image_names:
        images.append(cv2.resize(cv2.imread(osp.join(data_dir, 'images', each_img_name)), (input_shape[1], input_shape[0])))
        masks.append(np.expand_dims(cv2.resize(cv2.imread(osp.join(data_dir, 'mask_1c', each_img_name), cv2.IMREAD_GRAYSCALE), (input_shape[1], input_shape[0])), axis=2))
    data_X = np.array(images)
    data_mask = np.array(masks)
    print(type(data_X), data_X.shape)
    print(type(data_mask), data_mask.shape)

    seed = 818
    np.random.seed(seed)
    np.random.shuffle(data_X)
    np.random.seed(seed)
    np.random.shuffle(data_mask)
    if validate_dataset_name is not None:
        data_dir = osp.join(DATA_PATH, validate_dataset_name)
        test_image_names = os.listdir(osp.join(data_dir, 'images'))
        test_images = []
        test_masks = []
        for each_img_name in test_image_names:
            test_images.append(cv2.resize(cv2.imread(osp.join(data_dir, 'images', each_img_name)), (input_shape[1], input_shape[0])))
            test_masks.append(np.expand_dims(
                cv2.resize(cv2.imread(osp.join(data_dir, 'mask_1c', each_img_name), cv2.IMREAD_GRAYSCALE),
                           (input_shape[1], input_shape[0])), axis=2))

        test_data_X = np.array(test_images)
        test_data_mask = np.array(test_masks)

        np.random.seed(seed)
        np.random.shuffle(test_data_X)
        np.random.seed(seed)
        np.random.shuffle(test_data_mask)

        print("** Validation Data Loaded! **")
        print(type(test_data_X), test_data_X.shape)
        print(type(test_data_mask), test_data_mask.shape)
        return data_X / 255., data_mask, test_data_X / 255., test_data_mask

    return data_X / 255., data_mask, None, None

def check_data_folder():
    if not osp.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        os.mkdir(MNIST_DATA_PATH)
        input_data.read_data_sets(MNIST_DATA_PATH)

def check_folder(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path

def merge_images(images, size):
    _, h, w, c = images.shape
    if c in [3, 4]:
        # color channel
        img = np.zeros((h*size[0], w*size[1], c))
        for idx, each_image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h: j*h + h, i*w: i*w + w, :] = each_image
        return img
    elif c == 1:
        # single channel
        img = np.zeros((h*size[0], w*size[1]))
        for idx, each_image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h: j*h + h, i*w: i*w + w] = each_image[:, :, 0]
        return img
    else:
        raise ValueError("in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4")

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
    images = inverse_transform(images)
    image = np.squeeze(merge_images(images, size))
    return imsave(image_path, image)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.png'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

if __name__ == "__main__":
    load_ocr()
