import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import layers

from biutils.tfutils import selu


def get_initializer(act_fn):
    if act_fn == tf.nn.relu:
        return layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG')
    elif act_fn == selu:
        return layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
    else:  # xavier
        return layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG')


def softmax(logits, charmap_len):
    return tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, charmap_len])),tf.shape(logits))


def dcgan_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, out_fn=tf.nn.tanh, use_batchnorm=True):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    layers = []
    n_filter = 512
    img_res = 1
    l = tf.reshape(z, [-1, 1, 1, int(np.prod(z.get_shape()[1:]))])
    while img_res < img_shape[0] // 4:
        s = 2 if n_filter != 512 else 1 # first stride is smaller
        p = 'same' if n_filter != 512 else 'valid' # first stride is smaller
        l = tf.layers.conv2d_transpose(l, filters=n_filter, kernel_size=4, strides=s, padding=p, activation=None,
                                       kernel_regularizer=reg, data_format='channels_last',
                                       kernel_initializer=get_initializer(act_fn))
        if use_batchnorm:
            l = tf.layers.batch_normalization(l, training=True)
        l = act_fn(l)
        layers.append(l)
        img_res *= 2
        n_filter //= 2
    l = tf.layers.conv2d_transpose(l, filters=img_shape[-1], kernel_size=4, strides=2, padding='same', activation=out_fn,
                                   kernel_regularizer=reg, data_format='channels_last')
    layers.append(l)
    #print('GENERATOR:\n' + '\n'.join([str(l) for l in layers]))
    return layers


def my_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, out_fn=tf.nn.tanh, use_batchnorm=True):
    # NOTE: this is built after the figure in the DCGAN paper, but the
    # actual published code used a different arch
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None

    layers = []
    n_filter = 1024
    img_res = 4
    l = tf.layers.dense(z, img_res*img_res*n_filter, activation=tf.nn.relu, name="layer0")
    l = tf.reshape(l, [-1, img_res, img_res, n_filter])
    layers.append(l)
    i = 0
    while img_res < img_shape[0] // 2:
        with tf.name_scope('gen.%02d' % i):
            n_filter //= 2
            l = tf.layers.conv2d_transpose(l, filters=n_filter, kernel_size=5, strides=2, padding='same', activation=None,
                                           kernel_regularizer=reg, data_format='channels_last',
                                           kernel_initializer=get_initializer(act_fn))
            if use_batchnorm:
                l = tf.layers.batch_normalization(l, training=True)
            l = act_fn(l)
            layers.append(l)
            img_res *= 2
            i += 1
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


def ResBlock(name, inputs, n_hidden=512, activation=tf.nn.relu, l2_penalty=None, use_batchnorm=True, train=True):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    output = inputs
    output = activation(output)
    output = tf.layers.conv1d(output, n_hidden, 5, name=name+'.1', kernel_regularizer=reg, padding='same', data_format='channels_first')
    if use_batchnorm:
        output = tf.layers.batch_normalization(output, training=train, scale=False, epsilon=1e-5, momentum=0.9, name=name+".bn.1")
    output = activation(output)
    output = tf.layers.conv1d(output, n_hidden, 5, name=name+'.2', kernel_regularizer=reg, padding='same', data_format='channels_first')
    if use_batchnorm:
        output = tf.layers.batch_normalization(output, training=train, scale=False, epsilon=1e-5, momentum=0.9, name=name+".bn.2")
    return inputs + (0.3 * output)


def billion_word_generator(z, img_shape, l2_penalty, act_fn=tf.nn.relu, train=True):
    n_hidden = 512
    SEQ_LEN=img_shape[0]

    output = z
    output = tf.layers.dense(output, SEQ_LEN*n_hidden, name='Generator.Input')
    output = tf.layers.batch_normalization(output, scale=False, training=train, name='Generator.Input.bn')
    output = tf.reshape(output, [-1, n_hidden, SEQ_LEN])
    output = ResBlock('Generator.1', output, n_hidden, activation=act_fn, l2_penalty=l2_penalty, train=train, use_batchnorm=True)
    output = ResBlock('Generator.2', output, n_hidden, activation=act_fn, l2_penalty=l2_penalty, train=train, use_batchnorm=True)
    output = ResBlock('Generator.3', output, n_hidden, activation=act_fn, l2_penalty=l2_penalty, train=train, use_batchnorm=True)
    output = ResBlock('Generator.4', output, n_hidden, activation=act_fn, l2_penalty=l2_penalty, train=train, use_batchnorm=True)
    output = ResBlock('Generator.5', output, n_hidden, activation=act_fn, l2_penalty=l2_penalty, train=train, use_batchnorm=True)
    output = tf.layers.conv1d(output, img_shape[1], 1, name='Generator.Output', padding='same', data_format='channels_first')

    output = tf.transpose(output, [0, 2, 1])
    layers = [softmax(output, img_shape[1])]
    return layers



# We use the tanh as suggest in in LeCun's "Efficient Backprop", so that
# 0/1 aren't the saturation-endpoints of the output but are easily reachable.
# TODO: not sure if this is useful or not?
GENERATOR_OUT_FN = lambda x : 1.7159 * tf.nn.tanh(x * 2.0 / 3.0)

GENERATORS =  {
    'my':      lambda z, img_shape, l2p: my_generator(z, img_shape, l2p, act_fn=tf.nn.relu, out_fn=GENERATOR_OUT_FN),
    'my-tanh': lambda z, img_shape, l2p: my_generator(z, img_shape, l2p, act_fn=tf.nn.relu, out_fn=tf.nn.tanh),
    'dcgan':      lambda z, img_shape, l2p: dcgan_generator(z, img_shape, l2p, act_fn=tf.nn.relu, out_fn=GENERATOR_OUT_FN),
    'began':      lambda z, img_shape, l2p: began_generator(z, img_shape, l2p, out_fn=GENERATOR_OUT_FN),
    'billion_word': lambda z, img_shape, l2p: billion_word_generator(z, img_shape, l2p, act_fn=tf.nn.relu, train=True)
}
def create_generator(z, img_shape, l2_penalty, generator_type, batch_size=None):
    if generator_type not in GENERATORS:
        raise RuntimeError("unknown generator: %s" % generator)

    generator = GENERATORS[generator_type]
    with tf.variable_scope('generator'):
        layers = generator(z, img_shape, l2_penalty)
        out = tf.identity(layers[-1], name='out') # so we can refer to the output by a simple name
    return out


#-------------------------------------
def dcgan_discriminator(l, l2_penalty, n_output, n_hidden=64, act_fn=lambda x : tf.maximum(0.2*x, x), out_fn=None, use_batchnorm=True):
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    layers = []
    n_repeat = int(np.log2(int(l.get_shape()[1]))) - 2
    for i in range(n_repeat):
        l = tf.layers.conv2d(l, n_hidden, kernel_size=4, strides=2, padding='same', activation=None, kernel_regularizer=reg)
        if use_batchnorm:
            l = tf.layers.batch_normalization(l, training=True)
        l = act_fn(l)
        layers.append(l)
        n_hidden *= 2

    l = tf.layers.conv2d(l, n_output, kernel_size=4, strides=1, padding='valid', activation=None, kernel_regularizer=reg)
    l = tf.squeeze(l)
    layers.append(l)
    #print('DISCRIMINATOR:\n' + '\n'.join([str(l) for l in layers]))
    return layers



def my_discriminator(l, l2_penalty, n_output, n_hidden=64, act_fn=lambda x : tf.maximum(0.2*x, x), out_fn=None, use_batchnorm=True):
    # NOTE: this is built after the figure in the DCGAN paper, but the
    # actual published code used a different arch
    reg = tf.contrib.layers.l2_regularizer(l2_penalty) if l2_penalty > 0.0 else None
    layers = []
    n_repeat = int(np.log2(int(l.get_shape()[0]) / 8))
    for i in range(n_repeat):
        with tf.name_scope('disc.%02d' % i):
            l = tf.layers.conv2d(l, n_hidden, kernel_size=5, strides=2, padding='same', activation=None, kernel_regularizer=reg)
            if use_batchnorm:
                l = tf.layers.batch_normalization(l, training=True)
            l = act_fn(l)
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
    return layers


def billion_word_discriminator(x, l2_penalty, n_hidden=512, n_output=1, trainable=True):
    output = tf.transpose(x, [0,2,1])

    output = tf.layers.conv1d(output, n_hidden, 1, name='Discriminator.Input', padding='same', data_format='channels_first')
    output = ResBlock('Discriminator.1', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu, use_batchnorm=False)
    output = ResBlock('Discriminator.2', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu, use_batchnorm=False)
    output = ResBlock('Discriminator.3', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu, use_batchnorm=False)
    output = ResBlock('Discriminator.4', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu, use_batchnorm=False)
    output = ResBlock('Discriminator.5', output, n_hidden, l2_penalty=l2_penalty, activation=tf.nn.relu, use_batchnorm=False)
    output = tf.reshape(output, [-1, x.get_shape().as_list()[1] * n_hidden])
    output = tf.layers.dense(output, n_output, name='Generator.Output')
    layers = [output]
    return layers


DISCRIMINATORS =  {
        'began':      lambda a, l2p: began_autoencoder(a, n_output=1, n_hidden=128, l2_penalty=l2p),
        'dcgan':      lambda a, l2p: dcgan_discriminator(a, n_output=1, l2_penalty=l2p),
        'my':      lambda a, l2p: my_discriminator(a, n_output=1, l2_penalty=l2p),
        'my-big':  lambda a, l2p: my_discriminator(a, n_output=1, n_hidden=128, l2_penalty=l2p),
        'my-big2': lambda a, l2p: my_discriminator(a, n_output=1, n_hidden=256, l2_penalty=l2p),
        'billion_word': lambda a, l2p: billion_word_discriminator(a, n_output=1, n_hidden=512, l2_penalty=l2p)
    }
def create_discriminator(x, discriminator_type, l2_penalty, reuse_vars):
    out_fn = None
    disc_fn = DISCRIMINATORS[discriminator_type]
    if discriminator_type not in DISCRIMINATORS.keys():
        raise RuntimeError("Unknown discriminator: %s" % discriminator)

    with tf.variable_scope("discriminator", reuse=reuse_vars) as vs:
        disc_out = disc_fn(x, l2_penalty)[-1]

    return disc_out
