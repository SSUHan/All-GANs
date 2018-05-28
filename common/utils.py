import os.path as osp
import os
from tensorflow.examples.tutorials.mnist import input_data
import gzip
import numpy as np

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

    data = extract_data(osp.join(data_dir, '/train-images-idx3-ubyte.gz'), 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(osp.join(data_dir, '/train-images-idx3-ubyte.gz'), 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(osp.join(data_dir, '/train-images-idx3-ubyte.gz'), 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(osp.join(data_dir, '/train-images-idx3-ubyte.gz'), 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def check_data_folder():
    if not osp.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        os.mkdir(MNIST_DATA_PATH)
        input_data.read_data_sets(MNIST_DATA_PATH)