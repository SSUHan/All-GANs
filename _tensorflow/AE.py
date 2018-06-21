from _tensorflow.ops import *
from common.utils import save_images, check_folder
from _tensorflow.base import BASE
import tensorflow as tf
import time
import numpy as np
import os.path as osp

class AE(BASE):

    model_name = "AE"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point):

        super().__init__(sess, epoch, batch_size, z_dim, dataset_name,
                         checkpoint_dir, result_dir, log_dir, sample_point)


    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):

            net = lrelu(conv2d(x, 64, kernel_hw=(4, 4), stride_hw=(2, 2), name='en_conv1'))
            net = lrelu(batch_norm(conv2d(net, 128, kernel_hw=(4, 4), stride_hw=(2, 2), name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1]) # Flatten
            net = lrelu(batch_norm(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            net = linear(net, 100, scope='en_fc4')

            return net

    def decoder(self, z, is_training=True, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            net = tf.nn.relu(batch_norm(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(batch_norm(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                batch_norm(deconv2d(net, [self.batch_size, 14, 14, 64], kernel_hw=(4, 4), stride_hw=(2, 2), name='de_dc3'),
                           is_training=is_training,
                           scope='de_bn3')
            )
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], kernel_hw=(4, 4), stride_hw=(2, 2), name='de_dc4'))
            return out

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        """ Graph Input """
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_image')

        """ Loss Function """
        # encoding
        z = self.encoder(self.inputs, is_training=True, reuse=False)

        # decoding
        out = self.decoder(z, is_training=True, reuse=False)

        self.loss = tf.reduce_mean(tf.square(self.inputs - out)) # MSE

        """ Training """
        # optimizer
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.lr * 5, beta1=self.beta1).minimize(self.loss, var_list=t_vars)

        """ Testing """
        self.encode_vector = self.encoder(self.inputs, is_training=False, reuse=True)

        self.decode_images = self.decoder(self.encode_vector, is_training=False, reuse=True)

        """ Summary """
        self.loss_smy = tf.summary.scalar("loss", self.loss)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.test_images = self.data_X[:self.batch_size]

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


        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size: (idx+1)*self.batch_size]

                # update network
                _, summary_str, loss = self.sess.run([self.optim, self.loss_smy, self.loss],
                                                     feed_dict={self.inputs: batch_images})

                self.writer.add_summary(summary_str, counter)

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss))

                if np.mod(counter, self.sample_point) == 0:
                    samples = self.sess.run(self.decode_images,
                                            feed_dict={self.inputs: self.test_images})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(self.test_images[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                image_path=osp.join(
                                    check_folder(osp.join(check_folder(self.result_dir), self.model_dir)),
                                    self.model_name + '_origin.png'))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                image_path=osp.join(
                                    check_folder(osp.join(check_folder(self.result_dir), self.model_dir)),
                                    self.model_name + '_train{}_{}.png'.format(epoch, idx)))

                counter += 1

            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # TODO:show temporal results
            # self.visualize_results(epoch)

        # save mdoel for final step
        self.save(self.checkpoint_dir, counter)