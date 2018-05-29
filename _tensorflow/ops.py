import tensorflow as tf

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis"""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(_input, output_dim, kernel_height=5, kernel_width=5, stride_height=2, stride_width=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, _input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(_input, w, strides=[1, stride_height, stride_width, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def deconv2d(_input, output_shape, kernel_height=5, kernel_width=5, stride_height=2, stride_width=2,
             name='deconv2d', stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1], _input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(_input, w, output_shape=output_shape, strides=[1, stride_height, stride_width, 1])

        # Support for versions of Tensorflow befor 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(_input, w, output_shape=output_shape, strides=[1, stride_height, stride_width, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def linear(_input, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = _input.get_shape().as_list()

    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable("Matrix", [shape[1], output_size],
                                 tf.float32, tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(_input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(_input, matrix) + bias