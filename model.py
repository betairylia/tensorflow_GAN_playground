import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def generator(z, output_Dim, is_train=False, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 512, act=tf.identity, W_init=w_init, name='g_fc1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_fc1/bn')
        n = DenseLayer(n, 64 * int(output_Dim / 32) * int(output_Dim / 32), act=tf.identity, W_init=w_init, name='g_fc2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='g_fc2/bn')
        n = ReshapeLayer(n, [-1, int(output_Dim / 32), int(output_Dim / 32), 64], name='g_fc_reshape')

        temp = n

        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_first_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='g_n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add1')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/1')
        n = Subpixel
        Conv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/1')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/2')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/3')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/4')

        temp = n

        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n128s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c2/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n128s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_second_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='g_n128s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add2')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/4')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/5')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def generator2(z, output_Dim, is_train=False, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 512, act=tf.identity, W_init=w_init, name='g_fc1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_fc1/bn')
        n = DenseLayer(n, 64 * int(output_Dim / 32) * int(output_Dim / 32), act=tf.identity, W_init=w_init, name='g_fc2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_fc2/bn')
        n = ReshapeLayer(n, [-1, int(output_Dim / 32), int(output_Dim / 32), 64], name='g_fc_reshape')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n64s1/b/m')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_pixelshufflerx2/b/1')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_pixelshufflerx2/b/2')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/3')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_pixelshufflerx2/b/3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/4')

        temp = n

        for i in range(12):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n128s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c2/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n128s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_second_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n128s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add2')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/4')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_pixelshufflerx2/b/4')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/5')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_pixelshufflerx2/b/5')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def generator2_LN(z, output_Dim, is_train=False, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 512, act=tf.identity, W_init=w_init, name='g_fc1')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_fc1/ln')
        n = DenseLayer(n, 64 * int(output_Dim / 32) * int(output_Dim / 32), act=tf.identity, W_init=w_init, name='g_fc2')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_fc2/ln')
        n = ReshapeLayer(n, [-1, int(output_Dim / 32), int(output_Dim / 32), 64], name='g_fc_reshape')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s1/c/m')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_n64s1/l/m')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/1')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_pixelshufflerx2/l/1')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/2')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_pixelshufflerx2/l/2')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/3')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/3')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_pixelshufflerx2/l/3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n128s1/4')

        temp = n

        for i in range(12):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c1/%s' % i)
            nn = LayerNormLayer(nn, act=tf.nn.relu, reuse=reuse, name='g_n128s1/l1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c2/%s' % i)
            nn = LayerNormLayer(nn, act=tf.nn.relu, reuse=reuse, name='g_n128s1/l2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='g_second_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n128s1/c/m')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_n128s1/l/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add2')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/4')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_pixelshufflerx2/l/4')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='g_pixelshufflerx2/5')
        n = LayerNormLayer(n, act=tf.nn.relu, reuse=reuse, name='g_pixelshufflerx2/l/5')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def generator_shallow(z, output_Dim, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 64 * int(output_Dim / 32) * int(output_Dim / 32), act=tf.identity, W_init=w_init, name='g_fc1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_fc1/bn')
        n = ReshapeLayer(n, [-1, int(output_Dim / 32), int(output_Dim / 32), 64], name='g_fc_reshape')

        n = DeConv2d(n, 64, (3, 3), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n64s2/b/m')

        # /16

        n = DeConv2d(n, 48, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b/m')

        # /8

        n = DeConv2d(n, 48, (7, 7), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct2/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b2/m')

        temp = n

        # /4

        n = Conv2d(n, 48, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s1/c1/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s1/b1/m')

        n = Conv2d(n, 48, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s1/c2/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s1/b2/m')
        n = ElementwiseLayer([n, temp], tf.add, name='g_add2')

        n = DeConv2d(n, 48, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b3/m')

        # /2

        n = DeConv2d(n, 32, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n32s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n32s2/b/m')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def generator_shallowest(z, output_Dim, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("Generator", reuse=reuse) as vs:

        n = InputLayer(z, name='g_in')
        n = DenseLayer(n, 64 * int(output_Dim / 16) * int(output_Dim / 16), act=tf.identity, W_init=w_init, name='g_fc1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_fc1/bn')
        n = ReshapeLayer(n, [-1, int(output_Dim / 16), int(output_Dim / 16), 64], name='g_fc_reshape')

        n = DeConv2d(n, 64, (3, 3), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n64s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n64s2/b/m')

        # /8

        n = DeConv2d(n, 64, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b/m')

        # /4

        n = DeConv2d(n, 48, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct2/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b2/m')

        # /2

        n = DeConv2d(n, 48, (5, 5), strides=(2, 2), act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='g_n48s2/ct/m')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g_n48s2/b3/m')

        # /1

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n

def discriminator_deep(input_images, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x: tl.act.lrelu(x, 0.2)

        with tf.variable_scope("Discriminator", reuse=reuse) as vs:

            n = InputLayer(input_images, name='d_input/images')

            n = Conv2d(n, 64, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='d_n256s2/1')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/1')

            n = Conv2d(n, 64, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='d_n256s2/2')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/2')

            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d_n64s1/1')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/3')

            temp = n

            for i in range(8):
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n128s1/c1/%s' % i)
                nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_n128s1/l1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n128s1/c2/%s' % i)
                nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_n128s1/l2/%s' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name='d_first_residual_add/%s' % i)
                n = nn

            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d_n64s1/2')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/4')
            n = ElementwiseLayer([n, temp], tf.add, name='d_add1')

            n = Conv2d(n, 64, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='d_n64s2/1')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/5')

            n = Conv2d(n, 64, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='d_n64s2/2')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/6')

            n = Conv2d(n, 128, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='d_n64s2/3')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/7')

            temp = n

            for i in range(8):
                nn = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n64s1/c1/%s' % i)
                nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_n64s1/l1/%s' % i)
                nn = Conv2d(nn, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='d_n64s1/c2/%s' % i)
                nn = LayerNormLayer(nn, act=lrelu, reuse=reuse, name='d_n64s1/l2/%s' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name='d_second_residual_add/%s' % i)
                n = nn

            n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d_n64s1/3')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/8')
            n = ElementwiseLayer([n, temp], tf.add, name='d_add2')

            flatten = FlattenLayer(n, name='d_flatten')
            n = DenseLayer(flatten, n_units=128, act=tf.identity, W_init=w_init, name='d_fc1')
            n = LayerNormLayer(n, act=lrelu, reuse=reuse, name='d_ln/9')
            n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init, name='d_fc2')

            logits = n.outputs
            n.outputs = tf.nn.sigmoid(n.outputs)

            return n, logits

def discriminator_shallower(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 24
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='h2/l')
        net_h3 = Conv2d(net_h2, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = LayerNormLayer(net_h3, act=lrelu, reuse=reuse, name='h3/l')
        net_h4 = Conv2d(net_h3, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
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

def discriminator_shallowest(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 32
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='h2/l')
        net_h3 = Conv2d(net_h2, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = LayerNormLayer(net_h3, act=lrelu, reuse=reuse, name='h3/l')
        net_h4 = Conv2d(net_h3, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = LayerNormLayer(net_h4, act=lrelu, reuse=reuse, name='h4/l')

        net_ho = FlattenLayer(net_h4, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs

    return net_ho, logits

def discriminator_shallower_noLN(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 8
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')

        net_h1 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h2 = Conv2d(net_h1, df_dim * 2, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h3 = Conv2d(net_h2, df_dim * 4, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h4 = Conv2d(net_h3, df_dim * 8, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h5 = Conv2d(net_h4, df_dim * 16, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h6 = Conv2d(net_h5, df_dim * 8, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs

    return net_ho, logits

def discriminator_shallow(input_images, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 8
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, name='h0/c')
        net_h0 = LayerNormLayer(net_h0, act=lrelu, reuse=reuse, name='h0/l')

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = LayerNormLayer(net_h1, act=lrelu, reuse=reuse, name='h1/l')
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = LayerNormLayer(net_h2, act=lrelu, reuse=reuse, name='h2/l')
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = LayerNormLayer(net_h3, act=lrelu, reuse=reuse, name='h3/l')
        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = LayerNormLayer(net_h4, act=lrelu, reuse=reuse, name='h4/l')
        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = LayerNormLayer(net_h5, act=lrelu, reuse=reuse, name='h5/l')
        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
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
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits
