from common.utils import load_mnist, save_images, check_folder
import tensorflow as tf
from _tensorflow.ops import conv_cond_concat, lrelu, conv2d, batch_norm, linear, concat, deconv2d
import numpy as np
import os
import time
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from _tensorflow.base import BASE

class CGAN(BASE):
    model_name = "CGAN"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point):
        super().__init__(sess, epoch, batch_size, z_dim, dataset_name,
                         checkpoint_dir, result_dir, log_dir, sample_point)

    def discriminator(self, X, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            # merge image and label
            y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            X = conv_cond_concat(X, y)

            net = lrelu(conv2d(X, 64, (4, 4), (2, 2), name='d_conv1'))
            net = lrelu(batch_norm(conv2d(net, 128, (4, 4), (2, 2), name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(batch_norm(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            output_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(output_logit)

            return out, output_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            # merge noise and label
            z = concat([z, y], axis=1)

            net = tf.nn.relu(batch_norm(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(batch_norm(linear(net, 128*7*7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(batch_norm(
                deconv2d(net, [self.batch_size, 14, 14, 64], (4, 4), (2, 2), name='g_dc3'), is_training=is_training, scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], (4, 4), (2, 2), name='g_dc4'))

            return out

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.input_c_dim]
        batch_size = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [batch_size] + image_dims, name='real_images')

        # labels
        self.labels = tf.placeholder(tf.float32, [batch_size, self.y_dim], name='labels')

        # noises
        self.noises = tf.placeholder(tf.float32, [batch_size, self.z_dim], name='noises')

        """ Loss Function """
        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.labels, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.noises, self.labels, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.labels, is_training=True, reuse=True)


        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))
        )
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake))
        )

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake))
        )

        """ Training """
        # divide trainable variables into a group for D and a group for G
        trainable_vars = tf.trainable_variables()
        d_vars = [var for var in trainable_vars if 'd_' in var.name]
        g_vars = [var for var in trainable_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.lr*5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        """ Testing """
        # for test
        self.fake_images = self.generator(self.noises, self.labels, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_smy = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_smy = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_smy = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_smy = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.d_smy = tf.summary.merge([d_loss_real_smy, d_loss_smy])
        self.g_smy = tf.summary.merge([d_loss_fake_smy, g_loss_smy])

    def train(self):
        # initalize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_labels = self.data_y[:self.batch_size]

        start_epoch, start_batch_id, counter = self.before_train()

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_labels = self.data_y[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_smy, self.d_loss],
                                                       feed_dict={self.inputs: batch_images,
                                                                  self.labels: batch_labels,
                                                                  self.noises: batch_z})

                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_smy, self.g_loss],
                                                       feed_dict={self.labels: batch_labels,
                                                                  self.noises: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [{}] [{}/{}] time: {}, d_loss: {}, g_loss: {}"
                      .format(epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 step
                if np.mod(counter, self.sample_point) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.noises: self.sample_z,
                                                       self.labels: self.test_labels})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h*manifold_w, :, :, :], [manifold_h, manifold_w],
                                image_path=osp.join(check_folder(osp.join(check_folder(self.result_dir), self.model_dir)), self.model_name + '_train{}_{}.png'.format(epoch, idx)))

            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # TODO:show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)