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
from VAE_model import encoder_large_BN as encoder
from VAE_model import decoder_large_BN as decoder
from tensorlayer.prepro import *

batch_size = 64
nb_epochs = 2000
epochs_saveImg = 25
epochs_test = 50
iters_saveSample = 200
input_size = 128
latent_dim = 4
latent_lambda = 0.5

#Adam
lr = 1e-4 #alpha
beta1 = 0.9
beta2 = 0.99

# Read data
DATAPATH = sys.argv[1]
OUTPATH = DATAPATH[:-1] + "_output_VAE_Ex_LD" + str(latent_dim)
log_dir = 'logs/'

OUTPATH += "_" + sys.argv[2]
os.system("del /F /Q " + OUTPATH)
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

trainImgNum = min(50000, len(os.listdir(DATAPATH)))
batchCount = trainImgNum // batch_size

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def check_imgs_resize(x):
    if(x.shape[0] < input_size or x.shape[1] < input_size):
        x = imresize(x, [input_size, input_size])
    x = x / (255. / 2.)
    x = x - 1.
    return x

def force_resize(x):
    x = imresize(x, [input_size, input_size])
    x = x / (255. / 2.)
    x = x - 1.
    return x

def resize_n_crop(x, scale=1.2, is_random=True):
    dim_min = min(x.shape[0], x.shape[1])
    scale = float(input_size) * scale / float(dim_min)
    x = imresize(x, [int(scale * x.shape[0]), int(scale * x.shape[1])])
    x = crop(x, wrg=input_size, hrg=input_size, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=input_size, hrg=input_size, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def identity_img(x):
    return x

def train():
    ###==========LOAD DATA==========###
    train_img_list = sorted(tl.files.load_file_list(path = DATAPATH, regx = '.*.jpg', printable = False))[:50000]

    ###==========MODEL DEFINIATION==========###
    latent_input = tf.placeholder('float32', [batch_size, latent_dim], name = 'Input_noise')
    image_groundTruth = tf.placeholder('float32', [batch_size, input_size, input_size, 3], name = 'GroundTruth_Image')

    z_mean, z_stddev = encoder(image_groundTruth, latent_dim, is_train=True, reuse=False)
    random_samples = tf.random_normal([batch_size, latent_dim], 0, 1, dtype=tf.float32)
    z_final = z_mean + (z_stddev * random_samples)

    generated_images = decoder(z_final, is_train=True, reuse=False)

    generation_loss = tf.reduce_sum(tf.abs(generated_images - image_groundTruth))
    latent_loss = tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1)

    loss = tf.reduce_sum(generation_loss + latent_lambda * latent_loss)

    optim = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2, name='Adam_D').minimize(loss)

    test_generated_images = decoder(latent_input, is_train=False, reuse=True)

    ###========== SESS ==========###

    sess = tf.Session()

    writter = tf.summary.FileWriter(log_dir, sess.graph)
    tl.layers.initialize_global_variables(sess)

    # We just want the graph.
    writter.close()

    ###========== Initialize ==========###

    train_imgs = tl.vis.read_images(train_img_list, path = DATAPATH, n_threads = 16)

    ###========== Training ==========###

    print("[*] Training VAE")

    iter = 0
    step_time = 0
    for epoch in range(0, nb_epochs):

        epoch_time = time.time()
        n_iter = 0

        for idx in range(0, len(train_imgs), batch_size):

            #Throw the batch away!
            if(len(train_imgs) - idx < batch_size):
                break

            iter += 1

            gLoss, lLoss, tLoss = 0, 0, 0
            step_time = time.time()

            batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=resize_n_crop, is_random = False)
            # batch_imgs = tl.prepro.threading_data(train_imgs[idx:idx + batch_size], fn=force_resize)

            if(iter == 1):
                tl.vis.save_images(batch_imgs, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/Input_Sample_Real.png')

            if n_iter % iters_saveSample == 0:
                '''Get images from VAE'''
                out = sess.run(generated_images, {image_groundTruth: batch_imgs})
                tl.vis.save_images(out, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/VAE_Sample_%d.png' % n_iter)

            _, gLoss, lLoss, tLoss = sess.run([optim, generation_loss, latent_loss, loss], {image_groundTruth: batch_imgs})

            print("\tEpoch [%2d/%2d] %4d / %4d time: %4.4fs, gen_loss: %.8f latent_loss: %.8f tot_loss: %.8f" %
                  (epoch, nb_epochs, n_iter, batchCount, time.time() - step_time, gLoss, lLoss, tLoss))
            n_iter += 1

        log = "\n[*] Epoch: [%2d/%2d] time: %4.4fs\n" % (epoch, nb_epochs, time.time() - epoch_time)
        print(log)

        ## generate sample images
        if epoch % epochs_saveImg == 0:
            out = sess.run(generated_images, {image_groundTruth: batch_imgs})
            tl.vis.save_images(out, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/c_train_%d.png' % epoch)
            tl.vis.save_images(batch_imgs, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/c_train_%d_groundTruth.png' % epoch)
        
        ## generate test images
        if epoch % epochs_test == 0:
            _mean, _stddev = sess.run([z_mean, z_stddev], {image_groundTruth: batch_imgs})
            _random_sample = np.random.normal(0, 1, (batch_size, latent_dim))

            # interploate
            for _ii in range(int(batch_size / 8)):
                for _jj in range(8):
                    _alpha = (7.0 - float(_jj)) / 7.0
                    _mean[_ii * 8 + _jj, :] = _alpha * _mean[_ii * 8, :] + (1. - _alpha) * _mean[_ii * 8 + 7, :]
                    _stddev[_ii * 8 + _jj, :] = _alpha * _stddev[_ii * 8, :] + (1. - _alpha) * _stddev[_ii * 8 + 7, :]

            _final = _mean + (_stddev * _random_sample)

            out_inter = sess.run(test_generated_images, {latent_input: _final})
            out_raw = sess.run(test_generated_images, {latent_input: _random_sample})

            tl.vis.save_images(out_inter, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/a_INTER_test_%d.png' % epoch)
            tl.vis.save_images(out_raw, [8, int(math.ceil(float(batch_size) / 8.0))], OUTPATH + '/b_RAW_test_%d.png' % epoch)


train()
