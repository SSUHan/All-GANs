from common.utils import load_mnist

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
