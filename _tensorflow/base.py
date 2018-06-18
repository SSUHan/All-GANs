import os.path as osp
from common.utils import check_folder, load_mnist
import tensorflow as tf

class BASE(object):

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.sample_point = sample_point

        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            # params
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.y_dim = 10  # dimension of condition-vector (label)
            self.c_dim = 1  # dimension of color channel

            # train
            self.lr = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = int(len(self.data_X) / self.batch_size)

        else:
            raise NotImplementedError

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name,
            self.dataset_name,
            self.batch_size,
            self.z_dim
        )

    def save(self, checkpoint_dir, step):
        checkpoint_dir = osp.join(check_folder(osp.join(check_folder(checkpoint_dir), self.model_dir)), self.model_name)
        check_folder(checkpoint_dir)
        self.saver.save(self.sess, osp.join(checkpoint_dir, self.model_name + ".model"), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = osp.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = osp.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, osp.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0