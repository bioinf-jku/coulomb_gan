import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import layers


def get_initializer(act_fn):
    if act_fn == tf.nn.relu:
        return layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG')
    else:  # xavier
        return layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG')


def dcgan_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, out_fn=tf.nn.tanh):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None

    layers = []
    n_filter = 1024
    img_res = 4
    l = tf.layers.dense(z, img_res*img_res*n_filter, activation=tf.nn.relu, name="layer0")
    l = tf.reshape(l, [-1, img_res, img_res, n_filter])
    layers.append(l)

    while img_res < img_shape[0] // 2:
        n_filter //= 2
        l = tf.layers.conv2d_transpose(l, filters=n_filter, kernel_size=5, strides=2, padding='same', activation=None,
                                       kernel_regularizer=reg, data_format='channels_last',
                                       kernel_initializer=get_initializer(act_fn))
        l = tf.layers.batch_normalization(l, training=True)
        l = act_fn(l)
        layers.append(l)
        img_res *= 2
    l = tf.layers.conv2d_transpose(l, filters=img_shape[-1], kernel_size=5, strides=2, padding='same', activation=out_fn,
                                   kernel_regularizer=reg, data_format='channels_last')
    layers.append(l)
    return layers


def began_generator(z, img_shape, l2_penalty, act_fn=tf.nn.elu, out_fn=None):
    def scale(l, scaling):
        ls = l.get_shape().as_list()
        h, w = ls[1], ls[2]
        nh = int(h*scaling)
        nw = int(w*scaling)
        return tf.image.resize_nearest_neighbor(l, (nh, nw))

    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None

    layers = []
    n_repeat = 4
    n_hidden = 128
    l = tf.layers.dense(z, 8*8*n_hidden, activation=None, kernel_regularizer=reg)
    layers.append(l)
    l = tf.reshape(l, [-1, 8, 8, n_hidden])
    for idx in range(n_repeat):
        l = tf.layers.conv2d(l, n_hidden, 3, 1, activation=act_fn, padding='same',
                             kernel_regularizer=reg, kernel_initializer=get_initializer(act_fn))
        layers.append(l)
        l = tf.layers.conv2d(l, n_hidden, 3, 1, activation=act_fn, padding='same',
                             kernel_regularizer=reg, kernel_initializer=get_initializer(act_fn))
        layers.append(l)
        if idx < n_repeat - 1:
            l = scale(l, 2)
            layers.append(l)
    l = tf.layers.conv2d(l, img_shape[-1], 3, 1, activation=out_fn, padding='same', kernel_regularizer=reg)
    layers.append(l)
    return layers


def create_generator(z, img_shape, l2_penalty, generator_type, batch_size=None, out_fn=None):
    generators =  {
        'dcgan': lambda z: dcgan_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, out_fn=out_fn),
        'began': lambda z: began_generator(z, img_shape, l2_penalty, out_fn=out_fn),
    }

    if generator_type not in generators:
        raise RuntimeError("unknown generator: %s" % generator)

    generator = generators[generator_type]
    with tf.variable_scope('generator'):
        if 'fc' in generator_type:
            layers = generator(z)
            layers[-1] = tf.reshape(layers[-1], [-1] + img_shape)
        else:
            layers = generator(z)

        out = tf.identity(layers[-1], name='out') # so we can refer to the output by a simple name
    return out


#-------------------------------------


def dcgan_discriminator(l, l2_penalty, n_output, n_hidden=64, out_fn=None, reuse=False):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    with tf.variable_scope("dcgan", reuse=reuse):
        layers = []
        leaky_alpha = 0.2
        l = tf.layers.conv2d(l, n_hidden, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        l = tf.layers.batch_normalization(l, training=True)
        l = tf.maximum(leaky_alpha*l, l)
        layers.append(l)

        l = tf.layers.conv2d(l, n_hidden*2, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        l = tf.layers.batch_normalization(l, training=True)
        l = tf.maximum(leaky_alpha*l, l)
        layers.append(l)

        l = tf.layers.conv2d(l, n_hidden*4, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        l = tf.layers.batch_normalization(l, training=True)
        l = tf.maximum(leaky_alpha*l, l)
        layers.append(l)

        l = tf.layers.conv2d(l, n_hidden*8, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        l = tf.layers.batch_normalization(l, training=True)
        l = tf.maximum(leaky_alpha*l, l)
        layers.append(l)

        l = tf.reshape(l, [tf.shape(l)[0], np.prod(l.get_shape().as_list()[1:])])
        l = tf.layers.dense(l, n_output, activation=out_fn)
        layers.append(l)
    return layers


def create_discriminator(x, discriminator_type, l2_penalty, reuse_vars):
    out_fn = None
    discriminators =  {
        'dcgan': lambda a, r: dcgan_discriminator(a, n_output=1, l2_penalty=l2_penalty, out_fn=out_fn, reuse=r),
        'dcgan-big': lambda a, r: dcgan_discriminator(a, n_output=1, n_hidden=128, l2_penalty=l2_penalty, out_fn=out_fn, reuse=r),
        'dcgan-big2': lambda a, r: dcgan_discriminator(a, n_output=1, n_hidden=256, l2_penalty=l2_penalty, out_fn=out_fn, reuse=r),
    }

    disc_fn = discriminators[discriminator_type]
    if discriminator_type not in discriminators.keys():
        raise RuntimeError("Unknown discriminator: %s" % discriminator)

    # TODO: we should just deploy the 'reuse' here, instead of nesting 2 variable scopes
    with tf.variable_scope("discriminator") as vs:
        if 'fc' in discriminator_type:
            x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

        disc_out = disc_fn(x, reuse_vars)[-1]
        #layers_y = disc_fn(y, True)

    return disc_out
