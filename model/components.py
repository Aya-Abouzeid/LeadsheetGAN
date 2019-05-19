from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from model.libs.utils import *
from model.modules import *

class Model:
    def get_model_info(self, quiet=True):
        num_parameter_g = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.g_vars])
        num_parameter_d = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.d_vars])
        num_parameter_all = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.vars])

        if not quiet:
            print('# of parameters in G (generator)                 |', num_parameter_g)
            print('# of parameters in D (discriminator)             |', num_parameter_d)
            print('# of parameters in total                         |', num_parameter_all)

        return num_parameter_g, num_parameter_d, num_parameter_all

    def _build_optimizer(self, config):
        # self.print_vars(self.g_vars)
        with tf.variable_scope('Opt'):

            self.d_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.d_loss, var_list=self.d_vars)

            self.g_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.g_loss, var_list=self.g_vars)

    def print_vars(self, var_list):
        print('================================================')
        for v in var_list:
            print(v)




#######################################################################################################################
# NowBar
#######################################################################################################################

class Nowbar(Model):
    def _build_graph(self, config):
        self._build_encoder(config)
        self._build_generator(config)
        self._build_discriminator(config)
        self.print_vars(self.e_vars)
        self.g_vars = self.g_vars + self.e_vars

        self._build_optimizer(config)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def _build_encoder(self, config):
        with tf.variable_scope('E') as scope:
            if config.acc_idx is not None:
                    self.acc_track = tf.slice(self.x, [0, 0, 0, config.acc_idx], [-1, -1, -1, 1]) # take piano as condition
                    BE = BarEncoder()
                    self.nowbar = BE(in_tensor=self.acc_track, type_=config.type_)
            else:
                self.nowbar = None

            self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_generator(self, config):
        with tf.variable_scope('G') as scope:
            self.all_tracks = []

            for tidx in range(config.track_dim):
                if tidx is config.acc_idx:
                    tmp_track = self.acc_track
                else:
                    with tf.variable_scope(config.track_names[tidx]):
                        BG = BarGenerator(output_dim=self.output_dim)
                        ##print("nowbar",tf.shape(self.nowbar))
                        tmp_track = BG(in_tensor=self.z_final_list[tidx], nowbar=self.nowbar, type_=config.type_)

                self.all_tracks.append(tmp_track)

            self.prediction = tf.concat([t for t in self.all_tracks], 3)
            # print(self.prediction.get_shape())
            self.prediction_binary = to_binary_tf(self.prediction)
            self.prediction_chroma = to_chroma_tf(self.prediction_binary)

            self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            ## summary
            prediction_image = to_image_tf(self.prediction, config.colormap)
            self.summary_prediction_image = tf.summary.image('prediction/G', prediction_image,
                                                             max_outputs=10)

    def _build_discriminator(self, config):
        with tf.variable_scope('D') as scope:

            BD = BarDiscriminator()

            self.input_real = self.x
            self.input_fake = self.prediction

            _, self.D_real = BD(self.input_real, nowbar=self.nowbar, type_=config.type_, reuse=False)
            _, self.D_fake = BD(self.input_fake, nowbar=self.nowbar, type_=config.type_, reuse=True)

            ## compute gradient panelty
            # reshape data
            re_real = tf.reshape(self.input_real, [-1, config.output_w * config.output_h * config.track_dim])
            re_fake = tf.reshape(self.input_fake, [-1, config.output_w * config.output_h * config.track_dim])

            # sample alpha from uniform
            alpha = tf.random_uniform(
                                shape=[config.batch_size,1],
                                minval=0.,
                                maxval=1.)
            differences = re_fake - re_real
            interpolates = re_real + (alpha*differences)

            # feed interpolate into D
            X_hat = tf.reshape(interpolates, [-1, config.output_w, config.output_h, config.track_dim])
            _, self.D_hat = BD(X_hat, nowbar=self.nowbar, type_=config.type_, reuse=True)

            # compute gradients panelty
            gradients = tf.gradients(self.D_hat, [interpolates])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2) * config.lamda

            #loss
            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
            self.g_loss = -tf.reduce_mean(self.D_fake)
            self.d_loss += gradient_penalty

            self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

class NowbarHybrid(Nowbar):
    def __init__(self, config):
        with tf.variable_scope('NowbarHybrid'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_intra_dim = config.z_intra_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_intra = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim], name='z_intra')
            self.z_inter = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, config.track_dim], name='x')
            # to list
            self.z_final_list =  []

            for tidx in range(config.track_dim):
                z_intra =  tf.squeeze(tf.slice(self.z_intra, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                z_track = tf.concat([z_intra, self.z_inter], 1)
                self.z_final_list.append(z_track)

            self._build_graph(config)


#######################################################################################################################
# Temporal
#######################################################################################################################

