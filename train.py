#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:39:40 2019

@author: zhouchuansai
"""

import tensorflow as tf
from nets.predrnn import predrnn
import numpy as np
import os
from dataprocess import patchpro
from dataprocess import datasets_factory
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def train():
    batch_size = 8
    filters = [128, 64, 64, 64]
    kernel_size = 5
    inputlength = 10
    outputlength = 5
    channles = 1
    img_width = 64
    img_height = 64
    patch_size = 4
    num_layers = 4
    seqlength = inputlength + outputlength
    learning_rate = 0.001
    epochs = 30000
    test_interval = 100
    train_data_paths = ''
    valid_data_paths = ''
    save_model_path = ''


    train_input_handle, test_input_handle = datasets_factory.data_provider('mnist', train_data_paths,
                                                                           valid_data_paths, batch_size,
                                                                           img_width)
    
    # 使用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    channles = channles * patch_size * patch_size
    img_width = int(img_width / patch_size)
    img_height = int(img_height / patch_size)

    Predrnn = predrnn(batch_size, img_width, img_height, channles, filters, kernel_size, num_layers, 
                      seqlength, inputlength, patch_size)

    # 构建模型
    inputs = tf.keras.Input(batch_size=batch_size, shape=(seqlength, img_width, img_height,
                                                          channles))
    mask_true = tf.keras.Input(batch_size=batch_size, shape=(outputlength-1, img_width, 
                                                             img_height, channles))
    outputs = Predrnn(inputs, mask_true)
    model = tf.keras.Model(inputs=[inputs, mask_true], outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    

    delta = 0.00002
    base = 0.99998
    eta = 1
    
    for itr in range(epochs+1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        traindata = train_input_handle.get_batch()
        traindata = patchpro.getpatch(traindata, patch_size)

        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample((batch_size, outputlength-1))
        true_token = random_flip < eta

        ones = np.ones((img_width, img_height, channles))
        zeros = np.zeros((img_width, img_height, channles))
        mask_true_input = []
        for i in range(batch_size):
            for j in range(outputlength-1):
                if true_token[i, j]:
                    mask_true_input.append(ones)
                else:
                    mask_true_input.append(zeros)
        mask_true_input = np.array(mask_true_input, dtype='float32')
        mask_true_input = np.reshape(mask_true_input, (batch_size, outputlength-1, img_width, 
                                     img_height, channles))

        with tf.GradientTape() as tape:
            outputs = model([traindata, mask_true_input])
            loss_value = 2 * tf.nn.l2_loss(outputs - traindata[:,1:]) / (batch_size*(seqlength-1)*img_width*img_height*channles)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print('itr '+str(itr)+':')
        print('RMSE loss is: ', loss_value.numpy())

        # test
        if itr % test_interval == 0:
            print('-----------test------------')
            test_input_handle.begin(do_shuffle=False)
            batch_id = 0
            acc_test, rmse_test = [], []
            rmse_avg = 0
            acc_avg = 0
            #paths = os.path.join(save_result_path, str(itr))
            #if not os.path.exists(paths):
            #    os.makedirs(paths)
            for i in range(outputlength):
                acc_test.append(0)
                rmse_test.append(0)

            mask_true_input = np.zeros([batch_size, outputlength-1, img_width, img_height, channles],
                                       dtype='float32')

            while(test_input_handle.no_batch_left() == False):
                batch_id += 1
                validdata = test_input_handle.get_batch()
                validdata = patchpro.getpatch(validdata, patch_size)
                y_prediction = model([validdata, mask_true_input])
                y_pred = y_prediction[:, inputlength-1:]
                
            #    y_prediction = patchpro.patchback(y_prediction, patch_size)
            #    filename = paths+'/'+str(batch_id)+'_pred.npy'
            #    np.save(filename, y_prediction)

                for i in range(outputlength):
                    temp = (y_pred[:, i,:,:,0:patch_size*patch_size] - validdata[:, i+inputlength,:,:,0:patch_size*patch_size]).numpy()
                    rmse = np.sum(np.square(temp)) / (batch_size*img_width*img_height*channles)
                    rmse = np.sqrt(rmse)
                    rmse_test[i] += rmse
                    rmse_avg += rmse
                    acc = (temp < 2) * (temp > -2)
                    acc = np.sum(acc, dtype=float) / (batch_size*img_width*img_height*channles)
                    acc_test[i] += acc
                    acc_avg += acc


                test_input_handle.next()

            print('average RMSE of test is: ', rmse_avg/(batch_id*outputlength))
            print('average accuracy of test is: ', acc_avg/(batch_id*outputlength))
            print('\n')

            for i in range(outputlength):
                print('RMSE '+str(i)+' is: ', rmse_test[i]/batch_id)
                print('accuracy '+str(i)+' is: ', acc_test[i]/batch_id)

        # save model
        if itr % 10000 == 0:
            paths = os.path.join(save_model_path, str(itr))
            if not os.path.exists(paths):
                os.makedirs(paths)
            model.save(paths+'/my_model.h5')

        train_input_handle.next()
    
    
train()
