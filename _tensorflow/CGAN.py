from common.utils import load_mnist
import tensorflow as tf
from _tensorflow.ops import conv_cond_concat, lrelu, conv2d, batch_norm

class CGAN(object):
    model_name = "CGAN"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir

        if dataset_name == "mnist" or dataset_name == "fashion-mnist":
            # params
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim      # dimension of noise-vector
            self.y_dim = 10         # dimension of condition-vector (label)
            self.c_dim = 1          # dimension of color channel

            # train
            self.lr = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64    # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = int(len(self.data_X) / self.batch_size)

        else:
            raise NotImplementedError

    def discriminator(self, X, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_op_scope("discriminator", reus=reuse):
            # merge image and label
            y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            X = conv_cond_concat(X, y)

            net = lrelu(conv2d(X, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(batch_norm())

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        batch_size = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [None] + image_dims, name='real_images')

        # labels
        self.labels = tf.placeholder(tf.float32, [None, self.y_dim], name='labels')

        # noises
        self.noises = tf.placeholder(tf.float32, [None, self.z_dim], name='noises')

        """ Loss Function """
        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.labels, is_training=True, reuse=False)