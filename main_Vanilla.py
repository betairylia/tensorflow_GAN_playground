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
from model import generator_shallow as generator
from model import discriminator_shallower as discriminator
from tensorlayer.prepro import *

width = 1024
height = 1024
batch_size = 32
nb_epochs = 3000
epochs_saveImg = 1
input_crop_size = 128
z_dim = 64
GP_lambd = 10.0 # higher = more stable but slower convergence

d_iters = 1 # # of critic iterations / 1x genreator iterations
g_iters = 1 # do not use this
tot_iters = d_iters

#Adam
lr_init = 1e-4 #alpha
beta1 = 0.0
beta2 = 0.9

# Read data
DATAPATH = sys.argv[1]
OUTPATH = DATAPATH[:-1] + "_output"

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

    net_d, logits_real = discriminator(image_groundTruth, is_train = True, reuse = False)
    _, logits_fake = discriminator(net_g.outputs, is_train = True, reuse = True)

    # Normal GAN
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    # Wasserstein GAN
    # d_loss_real = - tf.reduce_mean(logits_real)
    # d_loss_fake = tf.reduce_mean(logits_fake)

    # d_loss = d_loss_real + d_loss_fake
    # em_estimate = - d_loss

    g_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    # g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    # alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval = 0.,maxval = 1.)
    # differences = net_g.outputs - image_groundTruth # This is different from MAGAN
    # interpolates = image_groundTruth + (alpha * differences)
    # # _, logits_inter = discriminator_deep(interpolates, is_train = True, reuse = True)
    # _, logits_inter = discriminator(interpolates, is_train = True, reuse = True)
    # gradients = tf.gradients(logits_inter, [interpolates])[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    # d_loss += GP_lambd * gradient_penalty

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


    ###========== Training ==========###
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

            errD, errG = 0, 0
            step_time = time.time()

            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=resize_n_crop)
            batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize)

            if(iter == 1):
                tl.vis.save_images(batch_imgs, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample.png')

            '''update D'''
            _errD, _ = sess.run([d_loss, d_optim], {z_input: batch_z, image_groundTruth: batch_imgs})

            # errD = errD + _errD / float(d_iters)
            # EMDist = EMDist + _EMDist / float(d_iters)
            errD += _errD

            print("D", end='')
            sys.stdout.flush()

            if(iter % d_iters == 0):
                '''update G'''
                _errG, _ = sess.run([g_loss, g_optim], {z_input: batch_z, image_groundTruth: batch_imgs})

                # errG = errG + _errG / float(g_iters)
                errG += _errG

                print("G", end='')
                sys.stdout.flush()

            print("\tEpoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f" %
                  (epoch, nb_epochs, n_iter, time.time() - step_time, errD, errG))
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
