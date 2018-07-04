import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def encoder(input_images, output_channels, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 32
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("encoder", reuse=reuse):
        n = InputLayer(input_images, name='input')

        n = Conv2d(n, df_dim * 1, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1') # 64
        n = Conv2d(n, df_dim * 2, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2') # 32

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3') # 32

        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 16
        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5') # 8

        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6') # 4

        flatten = FlattenLayer(n, name='reshape')

        n = DenseLayer(flatten, n_units = output_channels * 4, act=lrelu, W_init=w_init, name='fc')

        out_mean = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='mean')
        out_sdev = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='stddev')

    return out_mean.outputs, out_sdev.outputs

def decoder(z, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 32
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("decoder", reuse=reuse):
        n = InputLayer(z, name='input')

        n = DenseLayer(n, n_units = 4 * 4 * df_dim * 4, act=lrelu, W_init = w_init, name='fc')
        n = ReshapeLayer(n, [-1, 4, 4, df_dim * 4], name='reshape')

        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='convt1') # 8
        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='convt2') # 16
        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='convt3') # 32

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 32

        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='convt5') # 64
        n = DeConv2d(n, df_dim * 1, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='convt6') # 128

        out = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, b_init=b_init, name='conv7') # 128 (3c)

    return out.outputs

def encoder_BN(input_images, output_channels, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 32
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("encoder", reuse=reuse):
        n = InputLayer(input_images, name='input')

        n = Conv2d(n, df_dim * 1, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1') # 64
        n = Conv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv2') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv2/bn')

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv3') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv3/bn')

        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 16
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv4/bn')
        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv5') # 8
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv5/bn')
        
        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv6') # 4
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv6/bn')

        flatten = FlattenLayer(n, name='reshape')

        n = DenseLayer(flatten, n_units = output_channels * 4, act=tf.identity, W_init=w_init, name='fc')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='fc/bn')

        out_mean = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='mean')
        out_sdev = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='stddev')

    return out_mean.outputs, out_sdev.outputs

def decoder_BN(z, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 32
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("decoder", reuse=reuse):
        n = InputLayer(z, name='input')

        n = DenseLayer(n, n_units = 4 * 4 * df_dim * 4, act=tf.identity, W_init = w_init, name='fc')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='fc/bn')
        n = ReshapeLayer(n, [-1, 4, 4, df_dim * 4], name='reshape')

        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt1') # 8
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt1/bn')
        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt2') # 16
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt2/bn')
        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt3') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt3/bn')

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv4/bn')

        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt5') # 64
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt5/bn')
        n = DeConv2d(n, df_dim * 1, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt6') # 128
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt6/bn')

        out = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, b_init=b_init, name='conv7') # 128 (3c)

    return out.outputs

def encoder_large_BN(input_images, output_channels, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("encoder", reuse=reuse):
        n = InputLayer(input_images, name='input')

        n = Conv2d(n, df_dim * 1, (5, 5), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1') # 64
        n = Conv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv2') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv2/bn')

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv3') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv3/bn')

        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 16
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv4/bn')
        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv5') # 8
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv5/bn')
        
        n = Conv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv6') # 4
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv6/bn')

        # Res block
        temp = n

        for i in range(8):
            nn = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res%s/conv1' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='res%s/conv1/bn' % i)
            nn = Conv2d(nn, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res%s/conv2' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='res%s/conv2/bn' % i)

            nn = ElementwiseLayer([n, nn], tf.add, name='res%s/add' % i)
            n = nn
        
        n = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res/convOut')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='res/convOut/bn')
        n = ElementwiseLayer([n, temp], tf.add, name='res/addOut')

        n = Conv2d(n, df_dim * 8, (4, 4), (4, 4), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv7') # 1
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv7/bn')

        flatten = FlattenLayer(n, name='reshape')

        n = DenseLayer(flatten, n_units = output_channels * 4, act=tf.identity, W_init=w_init, name='fc')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='fc/bn')

        out_mean = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='mean')
        out_sdev = DenseLayer(n, n_units = output_channels, act=tf.nn.tanh, W_init=w_init, name='stddev')

    return out_mean.outputs, out_sdev.outputs

def decoder_large_BN(z, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("decoder", reuse=reuse):
        n = InputLayer(z, name='input')

        n = DenseLayer(n, n_units = 1 * 1 * df_dim * 8, act=tf.identity, W_init = w_init, name='fc')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='fc/bn')
        n = ReshapeLayer(n, [-1, 1, 1, df_dim * 8], name='reshape')

        n = DeConv2d(n, df_dim * 4, (4, 4), (4, 4), act=tf.identity, padding='VALID', W_init=w_init, b_init=b_init, name='res/convt0') # 4
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt0/bn')

        # Res block
        temp = n

        for i in range(8):
            nn = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res%s/conv1' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='res%s/conv1/bn' % i)
            nn = Conv2d(nn, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res%s/conv2' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='res%s/conv2/bn' % i)

            nn = ElementwiseLayer([n, nn], tf.add, name='res%s/add' % i)
            n = nn
        
        n = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='res/convOut')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='res/convOut/bn')
        n = ElementwiseLayer([n, temp], tf.add, name='res/addOut')

        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt1') # 8
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt1/bn')
        n = DeConv2d(n, df_dim * 4, (3, 3), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt2') # 16
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt2/bn')
        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt3') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt3/bn')

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv4') # 32
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='conv4/bn')

        n = DeConv2d(n, df_dim * 2, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt5') # 64
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt5/bn')
        n = DeConv2d(n, df_dim * 1, (5, 5), (2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='convt6') # 128
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='convt6/bn')

        out = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, b_init=b_init, name='conv7') # 128 (3c)

    return out.outputs
