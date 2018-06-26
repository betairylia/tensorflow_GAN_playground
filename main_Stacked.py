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
from model_Stacked import *
from tensorlayer.prepro import *

width = 1024
height = 1024
batch_size = 64
nb_epochs = 500
epochs_saveImg = 1
input_crop_size = 128
z_dim = 64
GP_lambd = 6.0 # higher = more stable but slower convergence
GP_count = 1
use_LP_instead_of_GP = True # Use LP instead of GP
noise_term_GP_LP_Sampling = 0.1

d_pretrain = 0 # epoches to pretrain the critic
d_iters = 5 # # of critic iterations / 1x genreator iterations
g_iters = 1 # do not use this
tot_iters = d_iters

# Discriminator: post generated is FAKE
# Related parameters
PostGenerated_Cache_size = 2000
Appearence_ratio = 0.8

#Adam
lr_init = 5e-4 #alpha
beta1 = 0
beta2 = 0.9

# Read data
DATAPATH = sys.argv[1]
OUTPATH = DATAPATH[:-1] + "_output_Stacked_AddPost"
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

def force_resize(x, size):
    x = imresize(x, size)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def identity_img(x):
    return x

def resize_n_crop(x, scale=1.2, is_random=True):
    dim_min = min(x.shape[0], x.shape[1])
    scale = float(input_crop_size) * scale / float(dim_min)
    x = imresize(x, [int(scale * x.shape[0]), int(scale * x.shape[1])])
    x = crop(x, wrg=input_crop_size, hrg=input_crop_size, is_random=is_random)
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
    train_img_list = sorted(tl.files.load_file_list(path = DATAPATH, regx = '.*.jpg', printable = False))[:50000]

    train_imgs = tl.vis.read_images(train_img_list, path = DATAPATH, n_threads = 16)

    post_generated_l1 = np.random.rand(PostGenerated_Cache_size, int(input_crop_size / 8), int(input_crop_size / 8), 3) * 2. - 1.
    post_generated_l2 = np.random.rand(PostGenerated_Cache_size, int(input_crop_size / 4), int(input_crop_size / 4), 3) * 2. - 1.
    post_generated_l3 = np.random.rand(PostGenerated_Cache_size, int(input_crop_size / 2), int(input_crop_size / 2), 3) * 2. - 1.
    post_generated_l4 = np.random.rand(PostGenerated_Cache_size, int(input_crop_size / 1), int(input_crop_size / 1), 3) * 2. - 1.

    post_generated = [post_generated_l1, post_generated_l2, post_generated_l3, post_generated_l4]

    ###==========MODEL DEFINIATION==========###
    z_input = tf.placeholder('float32', [batch_size, z_dim], name = 'Input_noise')

    image_groundTruth = [\
        tf.placeholder('float32', [batch_size, int(input_crop_size / 8), int(input_crop_size / 8), 3], name = 'GroundTruth_Image_L1'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 4), int(input_crop_size / 4), 3], name = 'GroundTruth_Image_L2'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 2), int(input_crop_size / 2), 3], name = 'GroundTruth_Image_L3'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 1), int(input_crop_size / 1), 3], name = 'GroundTruth_Image_L4')]

    net_g, net_g_l1, net_g_l2, net_g_l3, _ = generator_Stacked(z_input, input_crop_size, is_train = True, reuse = False)
    net_g_layers = [net_g_l1, net_g_l2, net_g_l3, net_g]

    discriminator_encoders = [discriminator_encoder_l1, discriminator_encoder_l2, discriminator_encoder_l3, discriminator_encoder_l4]

    logits_real = []
    logits_fake = []
    net_d = []

    for l in range(4):
        net_d_l, logits_real_l = discriminator_core(discriminator_encoders[l](image_groundTruth[l], is_train = True, reuse = False), is_train = True, reuse = (l > 0))
        _, logits_fake_l = discriminator_core(discriminator_encoders[l](net_g_layers[l].outputs, is_train = True, reuse = True), is_train = True, reuse = True)

        logits_real.append(logits_real_l)
        logits_fake.append(logits_fake_l)
        net_d.append(net_d_l)

    #Layer 1
    # net_d_l1, logits_real_l1 = discriminator_core(discriminator_encoder_l1(image_groundTruth[0], is_train = True, reuse = False), is_train = True, reuse = False)
    # _, logits_fake_l1 = discriminator_core(discriminator_encoder_l1(net_g_layers[0].outputs, is_train = True, reuse = True), is_train = True, reuse = True)
    #
    # #Layer 2
    # net_d_l2, logits_real_l2 = discriminator_core(discriminator_encoder_l2(image_groundTruth[1], is_train = True, reuse = False), is_train = True, reuse = True)
    # _, logits_fake_l2 = discriminator_core(discriminator_encoder_l2(net_g_layers[1].outputs, is_train = True, reuse = True), is_train = True, reuse = True)
    #
    # #Layer 3
    # net_d_l3, logits_real_l3 = discriminator_core(discriminator_encoder_l3(image_groundTruth[2], is_train = True, reuse = False), is_train = True, reuse = True)
    # _, logits_fake_l3 = discriminator_core(discriminator_encoder_l3(net_g_layers[2].outputs, is_train = True, reuse = True), is_train = True, reuse = True)
    #
    # #Layer 4
    # net_d_l4, logits_real_l4 = discriminator_core(discriminator_encoder_l4(image_groundTruth[3], is_train = True, reuse = False), is_train = True, reuse = True)
    # _, logits_fake_l4 = discriminator_core(discriminator_encoder_l4(net_g_layers[3].outputs, is_train = True, reuse = True), is_train = True, reuse = True)
    #
    # logits_real = [logits_real_l1, logits_real_l2, logits_real_l3, logits_real_l4]
    # logits_fake = [logits_fake_l1, logits_fake_l2, logits_fake_l3, logits_fake_l4]

    # Post generate part
    image_postGenerated = [\
        tf.placeholder('float32', [batch_size, int(input_crop_size / 8), int(input_crop_size / 8), 3], name = 'PostGenerate_Image_L1'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 4), int(input_crop_size / 4), 3], name = 'PostGenerate_Image_L2'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 2), int(input_crop_size / 2), 3], name = 'PostGenerate_Image_L3'), \
        tf.placeholder('float32', [batch_size, int(input_crop_size / 1), int(input_crop_size / 1), 3], name = 'PostGenerate_Image_L4')]

    logits_post = []

    for l in range(4):
        _, logits_postG_l = discriminator_core(discriminator_encoders[l](image_postGenerated[l], is_train = True, reuse = True), is_train = True, reuse = True)
        logits_post.append(logits_postG_l)

    #Wasserstein GAN
    d_loss_real = 0
    for l in range(4):
        d_loss_real += - tf.reduce_mean(logits_real[l])

    d_loss_fake = 0
    for l in range(4):
        d_loss_fake += tf.reduce_mean(logits_fake[l])

    d_loss_post = 0
    for l in range(4):
        d_loss_post += tf.reduce_mean(logits_post[l])

    d_loss = d_loss_real + d_loss_fake
    em_estimate = - d_loss

    d_loss = d_loss + d_loss_real + d_loss_post

    g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    for k in range(GP_count):
        for l in range(4):
            alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval = 0.,maxval = 1.)

            noise = tf.random_normal(shape=image_groundTruth[l].shape, stddev = noise_term_GP_LP_Sampling, mean = 0.)
            noised_outputs = net_g_layers[l].outputs + noise

            noise = tf.random_normal(shape=image_groundTruth[l].shape, stddev = noise_term_GP_LP_Sampling, mean = 0.)
            noised_groundTruth = image_groundTruth[l] + noise

            differences = noised_outputs - noised_groundTruth # This is different from MAGAN
            interpolates = noised_groundTruth + (alpha * differences)

            _, logits_inter = discriminator_core(discriminator_encoders[l](interpolates, is_train = True, reuse = True), is_train = True, reuse = True)
            # if i == 0:
            #     _, logits_inter = discriminator_core(discriminator_encoder_l1(interpolates, is_train = True, reuse = True), is_train = True, reuse = True)
            # if i == 1:
            #     _, logits_inter = discriminator_core(discriminator_encoder_l2(interpolates, is_train = True, reuse = True), is_train = True, reuse = True)
            # if i == 2:
            #     _, logits_inter = discriminator_core(discriminator_encoder_l3(interpolates, is_train = True, reuse = True), is_train = True, reuse = True)
            # if i == 3:
            #     _, logits_inter = discriminator_core(discriminator_encoder_l4(interpolates, is_train = True, reuse = True), is_train = True, reuse = True)

            gradients = tf.gradients(logits_inter, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
            gradient_penalty = 0

            if(use_LP_instead_of_GP):
                gradient_penalty = tf.reduce_mean(tf.nn.relu(slopes - 1.) ** 2)
            else:
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss += (1.0 / float(GP_count)) * GP_lambd * gradient_penalty * .25

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)
    d_vars = tl.layers.get_variables_with_name('Discriminator', True, True)

    # D_lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in d_vars if 'bias' not in v.name ]) * 0.001
    # d_loss += D_lossL2

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)

    _, net_g_l1_test, net_g_l2_test, net_g_l3_test, net_g_l4_test = generator_Stacked(z_input, input_crop_size, is_train = True, reuse = True)

    ###==========  ==========###
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.assign(lr_v, lr_init))

    ###========== Initialize ==========###


    ###========== Training ==========###
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
            batch_imgs_l1 = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize, size=[int(input_crop_size / 8), int(input_crop_size / 8)])
            batch_imgs_l2 = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize, size=[int(input_crop_size / 4), int(input_crop_size / 4)])
            batch_imgs_l3 = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize, size=[int(input_crop_size / 2), int(input_crop_size / 2)])
            batch_imgs_l4 = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize, size=[int(input_crop_size / 1), int(input_crop_size / 1)])

            batch_imgs = [batch_imgs_l1, batch_imgs_l2, batch_imgs_l3, batch_imgs_l4]

            post_gen_imgs = []
            for l in range(4):
                idx_postG = random.randint(0, PostGenerated_Cache_size - 1 - batch_size)
                post_imgs = tl.prepro.threading_data(post_generated[l][idx_postG:idx_postG + batch_size], fn = identity_img)
                post_gen_imgs.append(post_imgs)

            if(iter == 1):
                tl.vis.save_images(batch_imgs[0], [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_L1.png')
                tl.vis.save_images(batch_imgs[1], [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_L2.png')
                tl.vis.save_images(batch_imgs[2], [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_L3.png')
                tl.vis.save_images(batch_imgs[3], [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_L4.png')
                tl.vis.save_images(post_gen_imgs[3], [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_Post_Generate.png')

            feed_dict_img_groundTruth = dict(zip(image_groundTruth, batch_imgs))
            feed_dict_img_postGenerate = dict(zip(image_postGenerated, post_gen_imgs))
            feed_dict_z = {z_input: batch_z}

            # print({**feed_dict_z, **feed_dict_img_groundTruth, **feed_dict_img_postGenerate})

            '''update D'''
            _errD, _EMDist, _ = sess.run([d_loss, em_estimate, d_optim], {**feed_dict_z, **feed_dict_img_groundTruth, **feed_dict_img_postGenerate})

            # errD = errD + _errD / float(d_iters)
            # EMDist = EMDist + _EMDist / float(d_iters)
            errD += _errD
            EMDist += _EMDist

            print("D", end='')
            sys.stdout.flush()

            if(iter % d_iters == 0):
                '''update G'''
                _errG, _ = sess.run([g_loss, g_optim], {**feed_dict_z, **feed_dict_img_groundTruth, **feed_dict_img_postGenerate})

                # errG = errG + _errG / float(g_iters)
                errG += _errG

                print("G", end='')
                sys.stdout.flush()

                #Get post_generated
                out1, out2, out3, out4 = sess.run([net_g_l1_test.outputs, net_g_l2_test.outputs, net_g_l3_test.outputs, net_g_l4_test.outputs], {z_input: batch_z})

                out = [out1, out2, out3, out4]
                for _idx in range(batch_size):
                    for l in range(4):
                        if random.random() < Appearence_ratio:
                            # Randomly replace an image in cache
                            post_generated[l][random.randint(0, PostGenerated_Cache_size - 1), :, :, :] = out[l][_idx, :, :, :]

            print("\tEpoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f em_estimate: %.8f" %
                  (epoch, nb_epochs, n_iter, time.time() - step_time, errD, errG, EMDist))
            n_iter += 1

        log = "\n[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f\n" % (epoch, nb_epochs, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## generate sample images
        if (epoch != 0) and (epoch % epochs_saveImg == 0):
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            out1, out2, out3, out4 = sess.run([net_g_l1_test.outputs, net_g_l2_test.outputs, net_g_l3_test.outputs, net_g_l4_test.outputs], {z_input: batch_z})
            print(">>> Saving images...")
            tl.vis.save_images(out1, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/train_L1_%d.png' % epoch)
            tl.vis.save_images(out2, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/train_L2_%d.png' % epoch)
            tl.vis.save_images(out3, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/train_L3_%d.png' % epoch)
            tl.vis.save_images(out4, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/train_L4_%d.png' % epoch)

            idx_postG = random.randint(0, PostGenerated_Cache_size - 1 - batch_size)
            post_imgs = tl.prepro.threading_data(post_generated[3][idx_postG:idx_postG + batch_size], fn = identity_img)
            tl.vis.save_images(post_imgs, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Post_Generate_Cache_%d.png' % epoch)

train()
