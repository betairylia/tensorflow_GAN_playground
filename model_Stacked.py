import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

df_dim = 32

def generator_Stacked(z, output_Dim, is_train=False, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 64 * int(output_Dim / 32) * int(output_Dim / 32), act=tf.identity, W_init=w_init, name='g_fc1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_fc1/bn')
        n = ReshapeLayer(n, [-1, int(output_Dim / 32), int(output_Dim / 32), 64], name='g_fc_reshape')

        # 1/32 x 64

        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=lrelu, name='g_l1/ps1')

        # 1/16 x 16

        feature_count = 128

        n = Conv2d(n, 2 * feature_count, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l1/b1')

        # 1/16 x 256

        temp = n

        for i in range(4):
            nn = Conv2d(n, 2 * feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l1/c1/r%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l1/b1/r%s' % i)
            nn = Conv2d(nn, 2 * feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l1/c2/r%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l1/b2/r%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_l1_radd/%s' % i)
            n = nn

        n = Conv2d(n, 2 * feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l1/c/m')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_l1/add')

        n = Conv2d(n, 2 * feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_l1/c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=lrelu, name='g_l1/ps2')

        # 1/8 x 64

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='g_l1/out')
        layer1 = n

        #Layer 2

        n = Conv2d(n, feature_count, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l2/b1')

        temp = n

        for i in range(4):
            nn = Conv2d(n, feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l2/c1/r%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l2/b1/r%s' % i)
            nn = Conv2d(nn, feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l2/c2/r%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l2/b2/r%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_l2_radd/%s' % i)
            n = nn

        n = Conv2d(n, feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l2/c/m')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l2/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_l2/add')

        n = Conv2d(n, feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_l2/c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=lrelu, name='g_l2/ps2')

        # 1/4 x 32

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='g_l2/out')
        layer2 = n

        #Layer 3

        n = Conv2d(n, feature_count, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l3/b1')

        n = Conv2d(n, feature_count, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_l3/c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=lrelu, name='g_l3/ps2')

        # 1/2 x 32

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='g_l3/out')
        layer3 = n

        #Layer 4

        n = Conv2d(n, feature_count, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_l4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_l4/b1')

        n = Conv2d(n, feature_count // 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_l4/c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=lrelu, name='g_l4/ps2')

        # 1/2 x 32

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='g_l4/out')
        layer4 = n

        return n, layer1, layer2, layer3, layer4

def discriminator_core(input_encoded, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = input_encoded

        net_h4 = Conv2d(net_in, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = LayerNormLayer(net_h4, act=lrelu, reuse=reuse, name='h4/l')
        net_h5 = Conv2d(net_h4, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = LayerNormLayer(net_h5, act=lrelu, reuse=reuse, name='h5/l')
        net_h6 = Conv2d(net_h5, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = LayerNormLayer(net_h6, act=lrelu, reuse=reuse, name='h6/l')
        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = LayerNormLayer(net_h7, act=lrelu, reuse=reuse, name='h7/l')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = LayerNormLayer(net, act=lrelu, reuse=reuse, name='res/l')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = LayerNormLayer(net, act=lrelu, reuse=reuse, name='res/l2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = LayerNormLayer(net, act=lrelu, reuse=reuse, name='res/l3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs

    return net_ho, logits

def discriminator_encoder_l1(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim * 4, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l1h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='l1h1/l')

    return net_h1

def discriminator_encoder_l2(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim * 4, (7, 7), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l2h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='l2h1/l')

    return net_h1

def discriminator_encoder_l3(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim * 2, (7, 7), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l3h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='l3h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 4, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l3h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='l3h2/l')

    return net_h2

def discriminator_encoder_l4(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim * 2, (7, 7), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l4h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='l4h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 4, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l4h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='l4h2/l')
        net_h3 = Conv2d(net_h2, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='l4h3/c')
        net_h3 = LayerNormLayer(net_h3, act=lrelu, reuse=reuse, name='l4h3/l')

    return net_h3
