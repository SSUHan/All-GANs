import tensorflow as tf
from common.utils import load_mnist, save_images, check_folder
from _tensorflow.ops import lrelu, batch_norm, linear, conv2d, concat, deconv2d
import numpy as np
import os.path as osp
import time

class ACGAN(object):
    model_name = "ACGAN" # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.sample_point = sample_point

        if dataset_name == "mnist" or dataset_name == "fashion-mnist":
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

    def classifier(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)5c2s-(128)5c2s_BL-FC1024_BL-FC128_BL-FC12S’
        # All layers except the last two layers are shared by discriminator
        with tf.variable_scope("classifier", reuse=reuse):
            net = lrelu(batch_norm(linear(x, 128, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, self.y_dim, scope='c_fc2')
            out = tf.nn.softmax(out_logit)
            return out, out_logit

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope('discriminator', reuse=reuse):
            net = lrelu(conv2d(x, 64, (4, 4), (2, 2), name='d_conv1'))
            net = lrelu(batch_norm(conv2d(net, 128, (4, 4), (2, 2), name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(batch_norm(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope('generator', reuse=reuse):
            # merge noise and condition label
            z = concat([z, y], axis=1)

            net = tf.nn.relu(batch_norm(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(batch_norm(linear(net, 128*7*7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                batch_norm(deconv2d(net, [self.batch_size, 14, 14, 64], (4, 4), (2, 2), name='g_dc3'),
                           is_training=is_training, scope='g_bn3'))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], (4, 4), (2, 2), name='g_dc4'))
            return out

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        # labels
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        """ Loss Function """
        ## 1. GAN Loss
        # output of D for real images
        D_real, D_real_logit, input4classifier_real = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logit, input4classifier_fake = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real))
        )
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake)) # Note!
        )

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake)) # Note!
        )

        ## 2. infomation loss
        code_fake, code_fake_logit = self.classifier(input4classifier_fake, is_training=True, reuse=False)
        code_real, code_real_logit = self.classifier(input4classifier_real, is_training=True, reuse=True)

        # for fake samples
        c_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=code_fake_logit, labels=self.y)
        )

        # for real samples
        c_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=code_real_logit, labels=self.y)
        )

        self.c_loss = c_fake_loss + c_real_loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        c_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.lr * 5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
            self.c_optim = tf.train.AdamOptimizer(self.lr * 5, beta1=self.beta1).minimize(self.c_loss, var_list=c_vars)

        """ Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        # D
        d_loss_real_smy = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_smy = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_smy = tf.summary.scalar("d_loss", self.d_loss)

        # G
        g_loss_smy = tf.summary.scalar("g_loss", self.g_loss)

        # C
        c_loss_real_smy = tf.summary.scalar("c_loss_real", c_real_loss)
        c_loss_fake_smy = tf.summary.scalar("c_loss_fake", c_fake_loss)
        c_loss_smy = tf.summary.scalar("c_loss", self.c_loss)

        # final summary operations
        self.d_smy = tf.summary.merge([d_loss_real_smy, d_loss_smy])
        self.g_smy = tf.summary.merge([d_loss_fake_smy, g_loss_smy])
        self.c_smy = tf.summary.merge([c_loss_smy, c_loss_real_smy, c_loss_fake_smy])


    def train(self):
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_codes = self.data_y[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(osp.join(self.log_dir, self.model_name), self.sess.graph)

        # restore checkpoint if it exists:
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load Checkpoint Model Success!")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load Checkpoint Model Fail, Start Refresh")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size: (idx+1) * self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size: (idx+1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, smy_str_d, d_loss = self.sess.run([self.d_optim, self.d_smy, self.d_loss],
                                                   feed_dict={self.inputs: batch_images,
                                                              self.y: batch_codes,
                                                              self.z: batch_z})
                self.writer.add_summary(smy_str_d, counter)

                # update G & C network
                _, smy_str_g, g_loss, _, smy_str_c, c_loss = self.sess.run([self.g_optim, self.g_smy, self.g_loss,
                                                                            self.c_optim, self.c_smy, self.c_loss],
                                                                           feed_dict={self.inputs: batch_images,
                                                                                      self.y: batch_codes,
                                                                                      self.z: batch_z})
                self.writer.add_summary(smy_str_g, counter)
                self.writer.add_summary(smy_str_c, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every {self.sample_point} steps
                if np.mod(counter, self.sample_point) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z,
                                                       self.y: self.test_codes})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                image_path=osp.join(
                                    check_folder(osp.join(check_folder(self.result_dir), self.model_dir)),
                                    self.model_name + '_train{}_{}.png'.format(epoch, idx)))

            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

    def visualize_results(self):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name,
            self.dataset_name,
            self.batch_size,
            self.z_dim
        )

    def save(self, checkpoint_dir, step):
        checkpoint_dir = osp.join(checkpoint_dir, self.model_dir, self.model_name)
        check_folder(checkpoint_dir)
        self.saver.save(self.sess, osp.join(checkpoint_dir, self.model_name+".model"), global_step=step)

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