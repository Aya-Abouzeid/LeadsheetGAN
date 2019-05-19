'''
Model Configuration
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shutil import copyfile
import os
import SharedArray as sa
import tensorflow as tf
import glob

print('[*] config...')

# class Dataset:
TRACK_NAMES = ['mel', 'acc']
##TRACK_NAMES = ['bass', 'drums', 'guitar', 'piano', 'strings','chord']


def get_colormap():
    ##colormap = np.array([[1., 0., 0.],
    ##                     [0., 1., 0.],
    ##                     [0., 0., 1.],
    ##                     [1., .5, 0.],
    ##                     [0., .5, 1.],
    ##                     [0., 1., .5]])

    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.]])
    return tf.constant(colormap, dtype=tf.float32, name='colormap')

###########################################################################
# Training
###########################################################################

class TrainingConfig:
    is_eval = True
    batch_size = 32
    #batch_size = 64
    #batch_size = 32
    epoch = 100
    iter_to_save = 100
    sample_size = 250
    print_batch = True
    drum_filter = np.tile([1,0.3,0,0,0,0.3], 16)
    scale_mask = [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.]
    ##inter_pair = [(0,2), (0,3), (0,4), (0,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    inter_pair = [(0,1)]
    track_names = TRACK_NAMES
    track_dim = len(track_names)
    ##eval_map = np.array([
    ##                [1, 1, 1, 1, 1, 1],  # metric_is_empty_bar
    ##                [1, 1, 1, 1, 1, 1],  # metric_num_pitch_used
    ##                [1, 0, 1, 1, 1, 1],  # metric_too_short_note_ratio
    ##                [1, 0, 1, 1, 1, 1],  # metric_polyphonic_ratio
    ##                [1, 0, 1, 1, 1, 1],  # metric_in_scale
    ##                [0, 1, 0, 0, 0, 0],  # metric_drum_pattern
    ##                [1, 0, 1, 1, 1, 1]   # metric_num_chroma_used
    ##            ])


    eval_map = np.array([
                    [1,1],  # metric_is_empty_bar
                    [1,1],  # metric_num_pitch_used
                    [1,1],  # metric_too_short_note_ratio
                    [1,1],  # metric_polyphonic_ratio
                    [1,1],  # metric_in_scale
                    [0,0],  # metric_drum_pattern
                    [1,1]   # metric_num_chroma_used
                ])

    exp_name = 'exp'
    gpu_num = '0'


###########################################################################
# Model Config
###########################################################################

class ModelConfig:
    output_w = 96
    ##output_w = 48
    output_h = 84
    lamda = 10
    batch_size = 64
    #batch_size = 32
    beta1 = 0.5
    beta2 = 0.9
    lr = 2e-4
    is_bn = True
    colormap = get_colormap()


# nowbar
class NowBarHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 2
    acc_idx = 0
    z_inter_dim = 64
    z_intra_dim = 64
    output_dim = 1
    type_ = 0 # 0. for 96 ts perbar /1. for 48 ts per bar
    ##acc_output_w = 48 # chord sequence: 48, chroma sequence: 48, chroma vector: 4, chord vector: 4
    ##acc_output_h = 84 # chord sequence: 84, chroma sequence: 12, chroma vector:12, chord vector: 84

