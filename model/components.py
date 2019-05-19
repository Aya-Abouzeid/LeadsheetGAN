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

