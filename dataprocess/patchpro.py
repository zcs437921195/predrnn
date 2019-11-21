import numpy as np

# 该函数默认inputs是一个5维的numpy，[batch, seqlength, img_width, img_height, channels]

def getpatch(inputs, patch_size):
    batch_size = inputs.shape[0]
    seqlength = inputs.shape[1]
    img_width = inputs.shape[2]
    img_height = inputs.shape[3]
    channels = inputs.shape[4]
    patch_numpy = np.zeros([batch_size, seqlength, int(img_width/patch_size),
                            int(img_height/patch_size), channels*patch_size*patch_size])
    for i in range(batch_size):
        for j in range(seqlength):
            for k in range(channels):
                temp = inputs[i, j, :, :, k]
                temp = np.reshape(temp, [int(img_width/patch_size), int(img_height/patch_size),
                                         patch_size*patch_size])
                patch_numpy[i, j, :, :, k*patch_size*patch_size:(k+1)*patch_size*patch_size] = temp

    return patch_numpy


def patchback(inputs, patch_size):
    batch_size = inputs.shape[0]
    seqlength = inputs.shape[1]
    img_width = inputs.shape[2] * patch_size
    img_height = inputs.shape[3] * patch_size
    channels = int(inputs.shape[4] / (patch_size ** 2))
    img_numpy = np.zeros([batch_size, seqlength, img_width, img_height, channels])
    for i in range(batch_size):
        for j in range(seqlength):
            for k in range(channels):
                temp = inputs[i, j, :, :, k*patch_size*patch_size:(k+1)*patch_size*patch_size] 
                temp = np.reshape(temp, [img_width, img_height, 1])
                img_numpy[i, j, :, :, k] = temp[:,:,0]

    return img_numpy
