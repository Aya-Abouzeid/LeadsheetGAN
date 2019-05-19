from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
import SharedArray as sa

from model.core import *
from model.components import *
from input_data import *
from config import *

#assign GPU
if __name__ == '__main__':

    """ Create TensorFlow Session """

    t_config = TrainingConfig

    os.environ['CUDA_VISIBLE_DEVICES'] = t_config.gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        #path_x_train_phr =  'tra_X_phrase_all' # (50266, 384, 84, 5)
        path_x_train_phr =  'tra_X_phrs' # (5990, 384, 84, 2)

        # Temporal
            # hybrid
        ##t_config.exp_name = 'exps/temporal_hybrid'
        ##model = TemporalHybrid(TemporalHybridConfig)
        ##input_data = InputDataTemporalHybrid(model)
        ##input_data.add_data_sa(path_x_train_bar, 'train')
        
        # RNN
            # composer
        ##t_config.exp_name = 'exps/RNN_hybrid'
        ##t_config.exp_name = 'exps/NowBarRNN_hybrid'
        t_config.exp_name = 'exps/NowBar_hybrid'
        
        ##model = RNNHybrid(RNNHybridConfig) # RNN only from scratch
        ##model = RNNHybrid(NowBarRNNHybridConfig) # RNN + condition
        model = NowbarHybrid(NowBarHybridConfig)
        ##input_data = InputDataRNNHybrid(model)
        input_data = InputDataNowBarHybrid(model, ModelConfig.batch_size)
        input_data.add_data_sa(path_x_train_phr, 'train')
        
        musegan = MuseGAN(sess, t_config, model)
        ##musegan.load(musegan.dir_ckpt)
        musegan.train(input_data)

        ### load and generate samples ###
        # load pretrained model
        musegan.load(musegan.dir_ckpt)
    
        # add testing data
        path_x_test_phr = 'val_X_phrs'
        input_data.add_data_sa(path_x_test_phr, key='test')
        ##input_data.add_data_sa(path_x_train_phr, key='test')
    
        # generate samples
        musegan.gen_test(input_data, is_eval=True)


