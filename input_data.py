from __future__ import print_function
import numpy as np
import os
import SharedArray as sa
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class InputData:
    def __init__(self, model, batch_size=32):
        self.model = model # to get endpoint
        self.batch_size = batch_size
        self.z = dict()
        self.x = dict()

    def add_data(self, path_new, key='train'):
        self.x[key] = np.load(path_new)
        print('data size:', self.x[key].shape)

    def add_data_sa(self, path_new, key='train'):
        self.x[key] = sa.attach(path_new)
        print('data size:', self.x[key].shape)

    def add_data_np(self, data, key='train'):
        self.x[key] = data
        print('data size:', self.x[key].shape)

    def get_batch_num(self, key='train'):
        return len(self.x[key]) // self.batch_size

    def get_batch(self, idx=0, data_size=None, key='train'):
        data_size = self.batch_size if data_size is None else data_size
        st = self.batch_size*idx
        
        ##condition on chord/mel in testing part
        ##return [self.x[key][st+8]* 2. - 1.]*data_size ##
        
        return self.x[key][st:st+data_size] * 2. - 1.

    def get_rand_smaples(self, sample_size=64, key='train'):
        print(len(self.x[key]))
        random_idx = np.random.choice(len(self.x[key]), sample_size, replace=False)
        return self.x[key][random_idx]*2. - 1.

	#Fixed index and distribution.
    def gen_feed_dict(self, idx=0, data_size=None, key='train', z=None):
        batch_size = self.batch_size if data_size is None else data_size
        feed_dict = self.gen_z_dict(data_size=data_size, z=z)

        if key is not None:
            x = self.get_batch(idx, data_size, key)
            feed_dict[self.model.x] = x

        return feed_dict

#######################################################################################################################
# Image
#######################################################################################################################


#######################################################################################################################
# Music
#######################################################################################################################

# Nowbar
class InputDataNowBarHybrid(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
			#To be Changed.
            self.z['inter']=  np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
            self.z['intra'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
        z_dict = {self.model.z_intra: self.z['intra'], self.model.z_inter:self.z['inter']}
        return z_dict
