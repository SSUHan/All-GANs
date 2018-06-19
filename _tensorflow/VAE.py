from _tensorflow.base import BASE
import tensorflow as tf
from _tensorflow.ops import *
import common.prior_factory as prior
from common.utils import save_images, check_folder
import os.path as osp
import time
import numpy as np

class VAE(BASE):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, sample_point):

        super().__init__(sess, epoch, batch_size, z_dim, dataset_name,
                         checkpoint_dir, result_dir, log_dir, sample_point)

    # Gaussian Encoder
    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):

            net = lrelu(conv2d(x, 64, kernel_hw=(4, 4), stride_hw=(2, 2), name='en_conv1'))
            net = lrelu(batch_norm(conv2d(net, 128, kernel_hw=(4, 4), stride_hw=(2, 2), name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(batch_norm(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2*self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim] # gaussian_param.shape : [self.batch_size, 2*self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # and a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev

    # Bernoulli Decoder
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
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        """ Loss Function """
        # encoding
        self.mu, sigma = self.encoder(self.inputs, is_training=True, reuse=False)

        # sampling by re-parameterization technique for backpropagation algorithm
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out = self.decoder(z, is_training=True, reuse=False)

        # Any values less than clip_value_min are set to clip_value_min.
        # Any values greater than clip_value_max are set to clip_value_max
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out), [1, 2]) # cross entropy

        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence # ?
        # ELBO = self.neg_loglikelihood + self.KL_divergence

        self.loss = -ELBO # ?

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.lr * 5, beta1=self.beta1).minimize(self.loss, var_list=t_vars)


        """ Testing """
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        """ Summary """
        neg_loglikelihood_smy = tf.summary.scalar("neg_loglikelihood", self.neg_loglikelihood)
        kl_smy = tf.summary.scalar("kl_divergence", self.KL_divergence)
        loss_smy = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()


    def train(self):

        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

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


        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size: (idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, elbo_loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss,
                               self.neg_loglikelihood, self.KL_divergence],
                              feed_dict={self.inputs: batch_images,
                                         self.z: batch_z})

                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, elbo_loss: %.8f, nll_loss: %.8f, kl_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, elbo_loss, nll_loss, kl_loss))

                # save training results for every sample_point steps
                if np.mod(counter, self.sample_point) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
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
