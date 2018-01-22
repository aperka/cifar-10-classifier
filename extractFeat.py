#!/usr/bin/env python
#######encoding:utf-8
"""
@author:
@time:2017/3/18 14:33
"""
# Import the functions to calculate feature descriptions
from skimage import data, io, filters
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
# To read image file and save image feature descriptions
import os
import time
import glob
import matplotlib
from skimage import data, exposure

from config import *
import matplotlib.pyplot as plt

from six.moves import cPickle

hog_cases = {
        1: {"orientations": 9,
            "pixels_per_cell": [16, 16],
            "cells_per_block": [2, 2],
            "block_norm": 'L1',
            "visualize": True,
            "normalize": True
            },

        2: {"orientations": 9,
            "pixels_per_cell": [16, 16],
            "cells_per_block": [2, 2],
            "block_norm": 'L2-Hys',
            "visualize": True,
            "normalize": True
            },

        3: {"orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "block_norm": 'L2-Hys',
            "visualize": True,
            "normalize": True
            },

        4: {"orientations": 9,
            "pixels_per_cell": [16, 16],
            "cells_per_block": [1, 1],
            "block_norm": 'L2-Hys',
            "visualize": True,
            "normalize": True
            },

        5: {"orientations": 9,
            "pixels_per_cell": [16, 16],
            "cells_per_block": [2, 2],
            "block_norm": 'L2-Hys',
            "visualize": True,
            "normalize": True
            }
        }

def unpickle(file):
    f = open(file, 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()
    return datadict

def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        if childDir != 'test_batch':
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            train = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        else:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData


def getFeat(TrainData, TestData, ):

    if not os.path.exists('./data/features/test/'):
        os.makedirs('./data/features/test/')
    if not os.path.exists('./data/features/train/'):
        os.makedirs('./data/features/train/')

    fds = []
    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        #matplotlib.pyplot.imshow(image)
        gray = rgb2gray(image)/255.0
        #matplotlib.pyplot.imshow(gray)
        fd, hog_image = hog(gray, 16, (4,4), (1,1), 'L2-Hys', visualize, normalize)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        matplotlib.pyplot.imshow(hog_image, cmap=matplotlib.pyplot.cm.gray)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fds.append(fd)

    fd_path = os.path.join('./data/features/test/', fd_name)
    joblib.dump(fds, fd_path)
    print("Test features are extracted and saved.")

    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        fd, hog_image = hog(gray, 8, (4, 4), (1, 1), 'L2-Hys', visualize, normalize)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'

        fd_path = os.path.join('./data/features/train/', fd_name)
        joblib.dump(fd, fd_path)
    print("Train features are extracted and saved.")

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def save_features(file_path, dataset, hog_properties):
    fds = []
    for data in dataset:
        image = np.reshape(data[0].T, (32, 32, 3))
        #matplotlib.pyplot.imshow(image)
        gray = rgb2gray(image)/255.0
        #matplotlib.pyplot.imshow(gray)
        fd, hog_image = hog(gray, 16, (4,4), (1,1), 'L2-Hys', visualize, normalize)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        matplotlib.pyplot.imshow(hog_image, cmap=matplotlib.pyplot.cm.gray)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fds.append(fd)
    print(file_path)
    joblib.dump(fds, file_path)

if __name__ == '__main__':
    t0 = time.time()
    filePath = './cifar-10-batches-py'
    TrainData, TestData = getData(filePath)
    for hog_case in [1,2,3,4,5]:
        print(hog_case)
        save_features(os.path.join('data', str(hog_case)+'train.features'), TrainData, hog_cases[hog_case])
        save_features(os.path.join('data', str(hog_case)+'test.features'), TestData, hog_cases[hog_case])

    t1 = time.time()
    print("Features are extracted and saved.")
    print('The cast of time is:%f'%(t1-t0))