#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:54:40 2019

@author: zhouchuansai
"""

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import numpy as np
from tensorflow.keras import layers
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTM as cslstm

class predrnn(layers.Layer):
    def __init__(self, batch, img_width, img_height, channels, filters, kernel_size, num_layers, 
                 seqlength, inputlength, patch_size):
        super(predrnn, self).__init__()
        self.num_layers = num_layers
        self.seqlength = seqlength
        self.inputlength = inputlength
        self.filters = filters
        self.batch = batch
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.layer1 = cslstm(batch, img_width, img_height, channels, filters[0], kernel_size)
        self.layer2 = ghu(filters[0], kernel_size)
        self.layer3 = cslstm(batch, img_width, img_height, channels, filters[1], kernel_size)
        self.layer4 = cslstm(batch, img_width, img_height, channels, filters[2], kernel_size)
        self.layer5 = cslstm(batch, img_width, img_height, channels, filters[3], kernel_size)
        # modify here
        self.layer6 = layers.Conv2D(patch_size*patch_size, kernel_size=kernel_size, padding='same')
        #self.layer7 = layers.Conv2D(patch_size*patch_size, kernel_size=kernel_size, padding='same')
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_layers':self.num_layers, 'batch':self.batch, 'img_width':self.img_width,
                  'img_height':self.img_height, 'channels':self.channels, 'filters':self.filters,
                  'kernel_size':self.kernel_size, 'seqlength':self.seqlength, 
                  'inputlength':self.inputlength, 'patch_size':self.patch_size})
        return config


    def call(self, inputs, mask_true):

        # tf.enable_eager_execution()
        # tf.executing_eagerly()

        shape = tf.shape(inputs)
        # shape = [batch, seqlength, width, height, channels]
        m = tf.zeros([shape[0], shape[2], shape[3], self.filters[-1]])
        z = tf.zeros([shape[0], shape[2], shape[3], self.filters[0]])
        
        gen_images = []
        next_gen = []
        hidden = []
        cell = []
        for i in range(self.num_layers):
            hidden.append(tf.zeros([shape[0], shape[2], shape[3], self.filters[i]]))
            cell.append(tf.zeros([shape[0], shape[2], shape[3], self.filters[i]]))
        
        for t in range(self.seqlength-1):
            if t < self.inputlength:
                x = inputs[:, t, :, :, :]
            else:
                x = mask_true[:, t-self.inputlength]*inputs[:,t] + (1-mask_true[:, t-self.inputlength])*x_gen
                #x = mask_true[:, t-self.inputlength]*inputs[:,t] + (1-mask_true[:, t-self.inputlength])*hidden[3]

            hidden[0], cell[0], m = self.layer1(x, hidden[0], cell[0], m)
            z = self.layer2(hidden[0], z)
            hidden[1], cell[1], m = self.layer3(z, hidden[1], cell[1], m)
            
            hidden[2], cell[2], m = self.layer4(hidden[1], hidden[2], cell[2], m)
            hidden[3], cell[3], m = self.layer5(hidden[2], hidden[3], cell[3], m)
            
            #nextinput = self.layer6(hidden[3])
            x_gen = self.layer6(hidden[3])

            #nextinput = tf.expand_dims(hidden[3], 1)
            #next_gen.append(nextinput)
            #nextinput = tf.squeeze(nextinput, 1)

            #x_gen = self.layer7(nextinput)

            x_gen = tf.expand_dims(x_gen, 1)
            
            gen_images.append(x_gen)
            x_gen = tf.squeeze(x_gen, 1)
            
        #gen_images = np.concatenate(gen_images, axis=1)
        gen_images = tf.concat(gen_images, 1)
        #gen_images = tf.transpose(gen_images, [0,3,1,2])
        #next_gen = tf.concat(next_gen, 1)
        return gen_images
        
    #def loss_fn(self, gen_images, y_train):
    #    loss = 2 * tf.nn.l2_loss(gen_images - y_train[:,1:,:,:,0])
    #    return loss
        
        
        
        
