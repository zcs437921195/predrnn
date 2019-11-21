#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:17:57 2019

@author: zhouchuansai
"""

import tensorflow as tf
from tensorflow.keras import layers

class CausalLSTM(layers.Layer):
    def __init__(self, batch, img_width, img_height, inputchannels, filters, kernel_size):
        super(CausalLSTM, self).__init__()
        self.layer_h = layers.Conv2D(input_shape=(batch, img_width, img_height, inputchannels),
                                     filters=filters*4, kernel_size=kernel_size, padding='same')
        self.layer_c = layers.Conv2D(filters*3, kernel_size=kernel_size, padding='same')
        self.layer_m = layers.Conv2D(filters*3, kernel_size=kernel_size, padding='same')
        self.layer_x = layers.Conv2D(filters*7, kernel_size=kernel_size, padding='same')
        self.layer_cnew = layers.Conv2D(filters*4, kernel_size=kernel_size, padding='same')
        self.layer_mnew = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
        self.layer_cell = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
        self.forget_bias = 1.0
        
    # x, h, c, m分别代表论文当中缩写
    def call(self, x, h, c, m):
        
        h_cc = self.layer_h(h)
        i_h, g_h, f_h, o_h = tf.split(h_cc, 4, 3)
        
        c_cc = self.layer_c(c)
        i_c, g_c, f_c = tf.split(c_cc, 3, 3)
        
        m_cc = self.layer_m(m)
        i_m, f_m, m_m = tf.split(m_cc, 3, 3)
        
        x_cc = self.layer_x(x)
        i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = tf.split(x_cc, 7, 3)
        
        i = tf.sigmoid(i_x + i_h + i_c)
        f = tf.sigmoid(f_x + f_h + f_c + self.forget_bias)
        g = tf.tanh(g_x + g_h + g_c)
        
        c_new = f * c + i * g
        
        c2m = self.layer_cnew(c_new)
        i_c, g_c, f_c, o_c = tf.split(c2m, 4, 3)
        
        ii = tf.sigmoid(i_c + i_x_ + i_m)
        ff = tf.sigmoid(f_c + f_x_ + f_m + self.forget_bias)
        gg = tf.tanh(g_c + g_x_)
        
        m_new = ff * tf.tanh(m_m) + ii * gg
        
        o_m = self.layer_mnew(m_new)
        
        o = tf.tanh(o_x + o_h + o_c + o_m)
        
        cell = tf.concat([c_new, m_new], -1)
        cell = self.layer_cell(cell)
        
        h_new = o * tf.tanh(cell)
        
        return h_new, c_new, m_new
        
        

        
    
