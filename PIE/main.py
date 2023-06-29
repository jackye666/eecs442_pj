from pie_data import PIE
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pickle
import cv2
import sys

import xml.etree.ElementTree as ET
import numpy as np

from os.path import join, abspath, isfile, isdir
from os import makedirs, listdir
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import pylab
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import dataset

SEQUENCE_SIZE = 10
BATCH_SIZE = 4
pie_path = '.'
imdb = PIE(data_path=pie_path, regen_database=False, include_set="1,2,5", frinterval=1, downsample=1600)
database = imdb.generate_database()

if __name__ == '__main__':
    # first step: use this to get 1600x900 images
    # imdb.extract_and_save_images(extract_frame_type='annotated')
    # second step: use this to get pedestrian 128x128 images
    # dataset.extract_ped_images()
    # third step: all done, we read the data
    # num_available_peds = dataset.get_num_available_peds(sequence_size=SEQUENCE_SIZE)
    # print(num_available_peds)
    dataset.write_ped_intention_prob()
    ped_personal_images_intention, num_available_peds = \
        dataset.get_ped_images_and_intention_v_colab(sequence_size=SEQUENCE_SIZE, shuffle=True)
    # print(ped_personal_images_intention[0][1])
    # train_data = dataset.PedDataset(ped_personal_images_intention, data_range=(0, 50))
    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    # for i in range(10):
    #     for batch, label in tqdm(train_loader):
    #         print(batch.shape)
    # val_data = dataset.PedDataset(sequence_size=SEQUENCE_SIZE, data_range=(50, 90))
    # val_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    # test_data = dataset.PedDataset(sequence_size=SEQUENCE_SIZE, data_range=(90, 100))
    # test_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # ped_personal_images, ped_personal_bbox, ped_personal_intention_prob = dataset.get_ped_images_and_intention(
    #     sequence_size=15)

    # print(ped_personal_images['set01']['video_0001']['1_1_1'][0].shape)
    # print(ped_personal_bbox['set01']['video_0001']['1_1_1'][0])
    # todo: remember to regenerate the database first time >> regen_database=True

    # opts = {'fstride': 1,
    #           'sample_type': 'all',  # 'beh'
    #           'height_rng': [0, float('inf')],
    #           'squarify_ratio': 0,
    #           'data_split_type': 'default',  # kfold, random, default
    #           'seq_type': 'trajectory',
    #           'min_track_size': 15,
    #           'random_params': {'ratios': None,
    #                             'val_data': True,
    #                             'regen_data': False},
    #           'kfold_params': {'num_folds': 5, 'fold': 1}}
    # train = imdb.generate_data_trajectory_sequence('train', **opts)
    # val = imdb.generate_data_trajectory_sequence('val',**opts)
    # print(type(train), len(train), len(train['pid']))
    # print(train['image'][0])
    # print(type(val), len(val), len(val['pid']))
    # print(val['image'][0])
