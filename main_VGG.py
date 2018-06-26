import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import time
import math

import random
import sys
import os
from PIL import Image

from dataLoad import create_batch_Effecient
from model_VGGCritic import generator_shallow as generator
from model_VGGCritic import discriminator_vgg as discriminator
from model_VGGCritic import Vgg19_simple_api
from tensorlayer.prepro import *

width = 1024
height = 1024
batch_size = 48
nb_epochs = 1000
epochs_saveImg = 1
input_crop_size = 224
z_dim = 64
GP_lambd = 10.0 # higher = more stable but slower convergence
GP_count = 1
use_LP_instead_of_GP = True # Use LP instead of GP
noise_term_GP_LP_Sampling = 0.08

d_pretrain = 0 # epoches to pretrain the critic
d_iters = 10 # # of critic iterations / 1x genreator iterations
g_iters = 1 # do not use this
tot_iters = d_iters

#Adam
lr_init = 1e-4 #alpha
beta1 = 0
beta2 = 0.9

# Read data
DATAPATH = sys.argv[1]
OUTPATH = DATAPATH[:-1] + "_output_VGG"
if(use_LP_instead_of_GP):
    OUTPATH += "_LP"


os.system("del /f" + OUTPATH)
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

trainImgNum = len(os.listdir(DATAPATH))

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def check_imgs_resize(x):
    if(x.shape[0] < input_crop_size or x.shape[1] < input_crop_size):
        x = imresize(x, [input_crop_size, input_crop_size])
    x = x / (255. / 2.)
    x = x - 1.
    return x

def force_resize(x):
    x = imresize(x, [input_crop_size, input_crop_size])
    x = x / (255. / 2.)
    x = x - 1.
    return x

def resize_n_crop(x):
    dim_min = min(x.shape[0], x.shape[1])
    scale = float(input_crop_size) * 1.2 / float(dim_min)
    x = imresize(x, [int(scale * x.shape[0]), int(scale * x.shape[1])])
    x = crop(x, wrg=input_crop_size, hrg=input_crop_size, is_random=True)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=input_crop_size, hrg=input_crop_size, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def train():
    ###==========LOAD DATA==========###
    train_img_list = sorted(tl.files.load_file_list(path = DATAPATH, regx = '.*.jpg', printable = False))

    train_imgs = tl.vis.read_images(train_img_list, path = DATAPATH, n_threads = 16)

    ###==========MODEL DEFINIATION==========###
    z_input = tf.placeholder('float32', [batch_size, z_dim], name = 'Input_noise')
    image_groundTruth = tf.placeholder('float32', [batch_size, input_crop_size, input_crop_size, 3], name = 'GroundTruth_Image')

    net_g = generator(z_input, input_crop_size, is_train = True, reuse = False)

    net_vgg, fspace_real = Vgg19_simple_api((image_groundTruth + 1) / 2, reuse = False)
    net_d, logits_real = discriminator(fspace_real, is_train = True, reuse = False)

    _, fspace_fake = Vgg19_simple_api((net_g.outputs + 1) / 2, reuse = True)
    _, logits_fake = discriminator(fspace_fake, is_train = True, reuse = True)

    # Normal GAN
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    # d_loss = d_loss1 + d_loss2

    # Wasserstein GAN
    d_loss_real = - tf.reduce_mean(logits_real)
    d_loss_fake = tf.reduce_mean(logits_fake)

    d_loss = d_loss_real + d_loss_fake
    em_estimate = - d_loss

    # g_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    for i in range(GP_count):
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval = 0.,maxval = 1.)

        noise = tf.random_normal(shape=image_groundTruth.shape, stddev = -noise_term_GP_LP_Sampling, mean = 0.)
        noised_outputs = net_g.outputs + noise

        noise = tf.random_normal(shape=image_groundTruth.shape, stddev = -noise_term_GP_LP_Sampling, mean = 0.)
        noised_groundTruth = image_groundTruth + noise

        differences = noised_outputs - noised_groundTruth # This is different from MAGAN
        interpolates = noised_groundTruth + (alpha * differences)
        # _, logits_inter = discriminator_deep(interpolates, is_train = True, reuse = True)
        _, fspace_inter = Vgg19_simple_api((interpolates + 1) / 2, reuse = True)
        _, logits_inter = discriminator(fspace_inter, is_train = True, reuse = True)
        gradients = tf.gradients(logits_inter, [fspace_inter.outputs])[0]
        # gradients = tf.gradients(logits_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 0

        if(use_LP_instead_of_GP):
            gradient_penalty = tf.reduce_mean(tf.nn.relu(slopes - 1.) ** 2)
        else:
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss += (1.0 / float(GP_count)) * GP_lambd * gradient_penalty

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)
    d_vars = tl.layers.get_variables_with_name('Discriminator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)

    net_g_test = generator(z_input, input_crop_size, is_train = False, reuse = True)

    ###==========  ==========###
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.assign(lr_v, lr_init))

    ###========== Initialize ==========###

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###========== Training ==========###
    # pretrain
    print("[*] Pretrain D")
    for epoch in range(0, d_pretrain):
        epoch_time = time.time()
        n_iter = 0

        for idx in range(0, len(train_imgs), batch_size):

            #Throw the batch away!
            if(len(train_imgs) - idx < batch_size):
                break

            errD, D_real, D_fake = 0, 0, 0
            step_time = time.time()

            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=resize_n_crop)
            batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)

            '''update D'''
            errD, D_real, D_fake, _ = sess.run([d_loss, d_loss_real, d_loss_fake, d_optim], {z_input: batch_z, image_groundTruth: batch_imgs})

            print("D", end='')
            sys.stdout.flush()

            print("\tEpoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f d_real: %.8f d_fake: %.8f GP: %.8f" %
                  (epoch, d_pretrain, n_iter, time.time() - step_time, errD, -D_real, D_fake, errD - D_real - D_fake))
            n_iter += 1

    print("[*] Training GAN")

    iter = 0
    step_time = 0
    for epoch in range(0, nb_epochs):

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        errD, EMDist = 0, 0
        errG = 0

        for idx in range(0, len(train_imgs), batch_size):

            #Throw the batch away!
            if(len(train_imgs) - idx < batch_size):
                break

            iter += 1

            errD, EMDist, errG = 0, 0, 0
            step_time = time.time()

            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=resize_n_crop, is_random = False)
            batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)

            if(iter == 1):
                tl.vis.save_images(batch_imgs, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample.png')

            '''update D'''
            _errD, _EMDist, _ = sess.run([d_loss, em_estimate, d_optim], {z_input: batch_z, image_groundTruth: batch_imgs})

            # errD = errD + _errD / float(d_iters)
            # EMDist = EMDist + _EMDist / float(d_iters)
            errD += _errD
            EMDist += _EMDist

            print("D", end='')
            sys.stdout.flush()

            if(iter % d_iters == 0):
                '''update G'''
                _errG, _ = sess.run([g_loss, g_optim], {z_input: batch_z, image_groundTruth: batch_imgs})

                # errG = errG + _errG / float(g_iters)
                errG += _errG

                print("G", end='')
                sys.stdout.flush()

            print("\tEpoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f em_estimate: %.8f" %
                  (epoch, nb_epochs, n_iter, time.time() - step_time, errD, errG, EMDist))
            n_iter += 1

        log = "\n[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f\n" % (epoch, nb_epochs, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## generate sample images
        if (epoch != 0) and (epoch % epochs_saveImg == 0):
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            out = sess.run(net_g_test.outputs, {z_input: batch_z})
            print(">>> Saving images...")
            tl.vis.save_images(out, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/train_%d.png' % epoch)

train()
