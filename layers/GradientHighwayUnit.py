#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:36:51 2019

@author: zhouchuansai
"""

import tensorflow as tf
from tensorflow.keras import layers

class GHU(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(GHU, self).__init__()
        self.layer1 = layers.Conv2D(filters*2, kernel_size=kernel_size, padding='same')
        self.layer2 = layers.Conv2D(filters*2, kernel_size=kernel_size, padding='same')

    def call(self, x, z):
        z_concat = self.layer1(z)
        x_concat = self.layer2(x)
        p, u = tf.split(tf.add(x_concat, z_concat), 2, 3)
        p = tf.nn.tanh(p)
        u = tf.nn.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new
    
