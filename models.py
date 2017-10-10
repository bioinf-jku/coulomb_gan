import numpy as np
import tensorflow as tf
import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

from tensorflow.contrib import layers
BN_SCALE = False

def get_initializer(act_fn):
    if act_fn == tf.nn.relu:
        return layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG')
    else:  # xavier
        return layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG')

def softmax(logits, charmap_len):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, charmap_len])
        ),
        tf.shape(logits)
    )

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
    n_repeat = int(np.log2(img_shape[0] / 8)) + 1
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

class batch_norm(object):
    def __init__(self, scale=True, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
            self.scale = scale

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=self.scale,
                      is_training=train,
                      scope=self.name)

def ResBlock_bn(name, inputs, n_hidden=512, bn1=tf.identity, bn2=tf.identity, activation=tf.nn.relu, l2_penalty=None, train=True):
    output = inputs
    output = activation(output)
    output = bn1(lib.ops.conv1d.Conv1D(name+'.1', n_hidden, n_hidden, 5, output, weightnorm=l2_penalty), train=train)
    output = activation(output)
    output = bn2(lib.ops.conv1d.Conv1D(name+'.2', n_hidden, n_hidden, 5, output, weightnorm=l2_penalty), train=train)
    return inputs + (0.3 * output)

def billion_word_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, train=True):
    n_hidden = 512
    SEQ_LEN=img_shape[0]
    g_bn0 = batch_norm(scale=BN_SCALE, name='Generator.g_bn0')
    g_bn1 = batch_norm(scale=BN_SCALE, name='Generator.g_bn1')
    g_bn2 = batch_norm(scale=BN_SCALE, name='Generator.g_bn2')
    g_bn3 = batch_norm(scale=BN_SCALE, name='Generator.g_bn3')
    g_bn4 = batch_norm(scale=BN_SCALE, name='Generator.g_bn4')
    g_bn5 = batch_norm(scale=BN_SCALE, name='Generator.g_bn5')
    g_bn6 = batch_norm(scale=BN_SCALE, name='Generator.g_bn6')
    g_bn7 = batch_norm(scale=BN_SCALE, name='Generator.g_bn7')
    g_bn8 = batch_norm(scale=BN_SCALE, name='Generator.g_bn8')
    g_bn9 = batch_norm(scale=BN_SCALE, name='Generator.g_bn9')
    g_bn10 = batch_norm(scale=BN_SCALE, name='Generator.g_bn10')
    g_bn11 = batch_norm(scale=BN_SCALE, name='Generator.g_bn11')
    
    output = g_bn0(z, train=train)
    output = g_bn1(lib.ops.linear.Linear('Generator.Input', z.get_shape().as_list()[1], SEQ_LEN*n_hidden, output), train=train)
    output = tf.reshape(output, [-1, n_hidden, SEQ_LEN])
    output = ResBlock_bn('Generator.1', output, n_hidden, g_bn2, g_bn3, activation=act_fn, l2_penalty=l2_penalty, train=train)
    output = ResBlock_bn('Generator.2', output, n_hidden, g_bn4, g_bn5, activation=act_fn, l2_penalty=l2_penalty, train=train)
    output = ResBlock_bn('Generator.3', output, n_hidden, g_bn6, g_bn7, activation=act_fn, l2_penalty=l2_penalty, train=train)
    output = ResBlock_bn('Generator.4', output, n_hidden, g_bn8, g_bn9, activation=act_fn, l2_penalty=l2_penalty, train=train)
    output = ResBlock_bn('Generator.5', output, n_hidden, g_bn10, g_bn11, activation=act_fn, l2_penalty=l2_penalty, train=train)
    output = lib.ops.conv1d.Conv1D('Generator.Output', n_hidden, img_shape[1], 1, output)
    output = tf.transpose(output, [0, 2, 1])
    layers = [softmax(output, img_shape[1])]
    return layers

def create_generator(z, img_shape, l2_penalty, generator_type, batch_size=None, out_fn=None, train=True):
    generators =  {
        'dcgan': lambda z: dcgan_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, out_fn=out_fn),
        'began': lambda z: began_generator(z, img_shape, l2_penalty, out_fn=out_fn),
        'billion_word': lambda z: billion_word_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, train=True)
    }

    if generator_type not in generators:
        raise RuntimeError("unknown generator: %s" % generator)

    generator = generators[generator_type]
    with tf.variable_scope('generator') as scope:
        #TODO: This may lead to unexpected behavior, maybe I should change it
        if train == False:
            scope.reuse_variables()
        if 'fc' in generator_type:
            layers = generator(z)
            layers[-1] = tf.reshape(layers[-1], [-1] + img_shape)
        else:
            layers = generator(z)

        out = tf.identity(layers[-1], name='out') # so we can refer to the output by a simple name
    return out


#-------------------------------------


def dcgan_discriminator(l, l2_penalty, n_output, n_hidden=64, out_fn=None):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    layers = []
    leaky_alpha = 0.2
    n_repeat = int(np.log2(int(l.get_shape()[0]) / 8))
    for i in range(n_repeat):
        l = tf.layers.conv2d(l, n_hidden, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        l = tf.layers.batch_normalization(l, training=True)
        l = tf.maximum(leaky_alpha*l, l)
        layers.append(l)
        n_hidden *= 2

    l = tf.reshape(l, [tf.shape(l)[0], np.prod(l.get_shape().as_list()[1:])])
    l = tf.layers.dense(l, n_output, activation=out_fn)
    layers.append(l)
    return layers



def began_autoencoder(x, l2_penalty=0.0, n_hidden=128, n_output=64, out_fn=None, trainable=True):
    l = x
    layers = []
    act_fn = tf.nn.elu
    n_repeat = int(np.log2(int(l.get_shape()[0]) / 8)) + 1
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    init = get_initializer(act_fn)
    l = tf.layers.conv2d(l, n_hidden, 3, 1, activation=act_fn, padding='same', trainable=trainable, kernel_initializer=init)
    layers.append(l)
    for idx in range(n_repeat):
        n_channel = n_hidden * (idx + 1)
        l = tf.layers.conv2d(l, n_channel, 3, 1, activation=act_fn, padding='same',
                             trainable=trainable, kernel_regularizer=reg, kernel_initializer=init)
        layers.append(l)
        l = tf.layers.conv2d(l, n_channel, 3, 1, activation=act_fn, padding='same',
                             trainable=trainable, kernel_regularizer=reg, kernel_initializer=init)
        layers.append(l)
        if idx < n_repeat - 1:
            l = tf.layers.conv2d(l, n_channel, 3, 2, activation=act_fn, padding='same',
                                 trainable=trainable, kernel_regularizer=reg, kernel_initializer=init)
            layers.append(l)

    l = tf.reshape(l, [tf.shape(l)[0], np.prod(l.get_shape().as_list()[1:])])
    l = tf.layers.dense(l, n_output, activation=out_fn)
    layers.append(l)
    print('\n'.join([str(l) for l in layers]))
    return layers

def ResBlock(name, inputs, n_hidden=512, activation=tf.nn.relu, l2_penalty=None):
    output = inputs
    output = activation(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', n_hidden, n_hidden, 5, output, weightnorm=l2_penalty)
    output = activation(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', n_hidden, n_hidden, 5, output, weightnorm=l2_penalty)
    return inputs + (0.3 * output)

def billion_word_discriminator(x, l2_penalty, n_hidden=512, n_output=1, trainable=True):
    output = tf.transpose(x, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', x.get_shape().as_list()[2], n_hidden, 1, output)
    output = ResBlock('Discriminator.1', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu)
    output = ResBlock('Discriminator.2', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu)
    output = ResBlock('Discriminator.3', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu)
    output = ResBlock('Discriminator.4', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu)
    output = ResBlock('Discriminator.5', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu)
    output = tf.reshape(output, [-1, x.get_shape().as_list()[1] * n_hidden])
    output = lib.ops.linear.Linear('Discriminator.Output', x.get_shape().as_list()[1] * n_hidden, n_output, output)
    layers = [output]
    return layers

def create_discriminator(x, discriminator_type, l2_penalty, reuse_vars):
    out_fn = None
    discriminators =  {
        'began': lambda a: began_autoencoder(a, n_output=1, n_hidden=128, l2_penalty=l2_penalty, out_fn=out_fn),
        'dcgan': lambda a: dcgan_discriminator(a, n_output=1, l2_penalty=l2_penalty, out_fn=out_fn),
        'dcgan-big': lambda a: dcgan_discriminator(a, n_output=1, n_hidden=128, l2_penalty=l2_penalty, out_fn=out_fn),
        'dcgan-big2': lambda a: dcgan_discriminator(a, n_output=1, n_hidden=256, l2_penalty=l2_penalty, out_fn=out_fn),
        'billion_word': lambda a: billion_word_discriminator(a, n_output=1, n_hidden=512, l2_penalty=l2_penalty)
    }

    disc_fn = discriminators[discriminator_type]
    if discriminator_type not in discriminators.keys():
        raise RuntimeError("Unknown discriminator: %s" % discriminator)

    with tf.variable_scope("discriminator", reuse=reuse_vars) as vs:
        disc_out = disc_fn(x)[-1]

    return disc_out
