import os.path as osp
from common.utils import check_folder, load_mnist, load_ocr
import tensorflow as tf


class BASE(object):
    model_name = "BASE"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point, model_version):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.sample_point = sample_point
        self.model_verison = model_version

        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            # params
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.y_dim = 10  # dimension of condition-vector (label)
            self.input_c_dim = 1  # dimension of color channel

            # train
            self.lr = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = int(len(self.data_X) / self.batch_size)

        elif dataset_name == "ocr_eng_vertical_1000":
            self.input_height = 120
            self.input_width = 16
            self.input_c_dim = 3  # rgb

            self.output_height = 120
            self.output_width = 16
            self.output_c_dim = 1  # for mask gen

            self.z_dim = z_dim
            self.y_dim = 10

            # train
            self.lr = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 9  # number of generated images to be saved

            # load ocr
            self.data_X, self.data_mask, self.test_data_X, self.test_data_mask = load_ocr(self.dataset_name,
                                                                                          input_shape=(self.input_height, self.input_width, self.input_c_dim),
                                                                                          validate_dataset_name=dataset_name + '_test',
                                                                                          sample_num=self.sample_num)

            # get number of batches for a single epoch
            self.num_batches = int(len(self.data_X) / self.batch_size)

        else:
            raise NotImplementedError

    def before_train(self):

        # summary writer
        self.writer = tf.summary.FileWriter(osp.join(self.log_dir, self.model_name), self.sess.graph)

        # saver to save model
        self.saver = tf.train.Saver()

        # restore checkpoint if it exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load Success..!")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 0
            print(" [!] Load Fail, Start New Model..!")

        return start_epoch, start_batch_id, counter

    @property
    def model_dir(self):
        return "{}_v{}_{}_{}_{}".format(
            self.model_name,
            self.model_verison,
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
