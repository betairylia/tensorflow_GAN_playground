import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def generator_decoder(input, input_channels=128, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator/Decoder", reuse=reuse) as vs:

        n = InputLayer(input, name='g_in')
        n = DeConv2d(n, 96, (3, 3), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s2/ct/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dec/l1')

        # /8

        n = DeConv2d(n, 64, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dec/l2')

        # /4

        n = DeConv2d(n, 64, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct2/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dec/l3')

        # /2

        n = DeConv2d(n, 48, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dec/l4')

        # /1

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def generator_dense(z, output_Dim, output_channels=128, is_train=False, reuse=False, resLenth=16):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator/Dense", reuse=reuse) as vs:
        n = InputLayer(z, name='g_in')

        n = DenseLayer(n, 512, act=tf.identity, W_init=w_init, name='g_fc1')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dense/fc1/l')
        n = DenseLayer(n, 32 * int(output_Dim / 16) * int(output_Dim / 16), act=tf.identity, W_init=w_init, name='g_fc2')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dense/fc2/l')
        n = ReshapeLayer(n, [-1, int(output_Dim / 16), int(output_Dim / 16), 32], name='g_fc_reshape')

        n = Conv2d(n, output_channels, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_conv_reshape')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dense/c1/l')

        temp = n

        for i in range(resLenth):
            nn = Conv2d(n, output_channels, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c1/%s' % i)
            nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='g_dense/resc1/l/%s' % i)
            nn = Conv2d(nn, output_channels, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c2/%s' % i)
            nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='g_dense/resc2/l/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, output_channels, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='g_dense/cf/l')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add1')

        return n

def discriminator_encoder(input_images, output_channels=128, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = output_channels / 4
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator/Encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='h2/l')
        net_h3 = Conv2d(net_h2, df_dim * 3, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = LayerNormLayer(net_h3, act=lrelu, reuse=reuse, name='h3/l')
        net_h4 = Conv2d(net_h3, output_channels, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = LayerNormLayer(net_h4, act=lrelu, reuse=reuse, name='h4/l')

    return net_h4

def discriminator_dense(input, input_channels=128, is_train=False, reuse=False, resLenth = 16):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator/Dense", reuse=reuse) as vs:
        n = InputLayer(input, name='d_dense_in')

        temp = n

        for i in range(resLenth):
            nn = Conv2d(n, input_channels, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n64s1/c1/%s' % i)
            nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_dense/resc1/l/%s' % i)
            nn = Conv2d(nn, input_channels, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n64s1/c2/%s' % i)
            nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_dense/resc2/l/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='d_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, input_channels, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n64s1/c/m')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_dense/cf/l')
        n = ElementwiseLayer([n, temp], tf.add, name='d_add1')

        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_conv_reshape')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_dense/c1/l')

        flatten = FlattenLayer(n, name='d_flatten')
        n = DenseLayer(flatten, n_units=128, act=tf.identity, W_init=w_init, name='d_fc1')
        n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/9')
        n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init, name='d_fc2')

        logits = n.outputs

        return n, logits
