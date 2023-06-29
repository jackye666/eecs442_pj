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
import os
import math
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import random
import pandas as pd

NUM_IMAGES_PER_ROUND = 3
pie_path = '.'
imdb = PIE(data_path=pie_path, regen_database=False, include_set="1,2,5", frinterval=1, downsample=1600)
database = imdb.generate_database()


def get_ped_images_and_intention_v_colab(sequence_size=10, shuffle=True):
    ped_personal_intention_prob = pd.read_csv("ped_personal_intention_prob.csv")
    ped_personal_images_intention = []
    start = False
    for root, dirs, files in os.walk(imdb._input_path):
        if start:
            temp_dir = []
            index_file = 0
            sequence_index = 0
            num_files = len(os.listdir(root))
            if num_files < sequence_size:
                continue
            for file in files:
                if index_file % (
                        int(math.floor(num_files / sequence_size))) != 0 or sequence_index >= sequence_size:
                    index_file += 1
                    continue
                else:
                    sequence_index += 1
                    temp_dir.append(
                        (torch.from_numpy(cv2.imread(join(root, file)).astype("f").transpose(2, 0,
                                                                                             1) / 128.0 - 1.0)))
                index_file += 1
            # print(ped_personal_intention_prob[root[8:]])
            ped_personal_images_intention.append((temp_dir, float(ped_personal_intention_prob[root[8:]])))
        else:
            start = True
    num_available_peds = len(ped_personal_images_intention)
    # print(num_available_peds)
    if shuffle:
        random.shuffle(ped_personal_images_intention)
    return ped_personal_images_intention, num_available_peds


def write_ped_intention_prob():
    ped_personal_intention_prob = {}
    for sid in sorted(database):
        for vid in sorted(database[sid]):
            for pid in database[sid][vid]['ped_annotations'].keys():
                ped_personal_intention_prob[pid] = [database[sid][vid]['ped_annotations'][pid]['attributes']['intention_prob']]
    dataframe = pd.DataFrame(ped_personal_intention_prob)
    dataframe.to_csv("ped_personal_intention_prob.csv", index=False, sep=',')


def get_num_available_peds(sequence_size=10):
    num_peds = 0
    # iterate every pedestrian
    for sid in sorted(database):
        for vid in sorted(database[sid]):
            for pid in database[sid][vid]['ped_annotations'].keys():
                # print(pid)
                input_subpath = join(imdb._input_path, pid)
                num_files = len(os.listdir(input_subpath))
                if num_files < sequence_size:
                    continue
                num_peds += 1
    return num_peds


class PedDataset(Dataset):
    def __init__(self, ped_personal_images_intention, data_range=(0, 1)):
        self.dataset = ped_personal_images_intention[data_range[0]:data_range[1]]
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_seq, label = self.dataset[index]
        seq_tensor = torch.stack(img_seq)
        label = int(label > 0.5)

        return torch.FloatTensor(seq_tensor), torch.tensor(label)


def get_ped_images_and_intention(sequence_size=10, shuffle=True):
    # max_sequence_size = 12 limited by 1_1_14
    # we jump over the ped if it doesn't meet the sequence size

    # Below is to get every pedestrian's corresponding images and intention prob:
    ped_personal_image = {}
    ped_personal_bbox = {}
    ped_personal_intention_prob = {}
    # min_file_num = 1000000000000000
    # iterate every pedestrian
    for sid in sorted(database):
        # print(sid)
        ped_personal_image[sid] = {}
        ped_personal_bbox[sid] = {}
        ped_personal_intention_prob[sid] = {}
        for vid in sorted(database[sid]):
            # print(vid)
            ped_personal_image[sid][vid] = {}
            ped_personal_bbox[sid][vid] = {}
            ped_personal_intention_prob[sid][vid] = {}
            for pid in database[sid][vid]['ped_annotations'].keys():
                # print(pid)
                input_subpath = join(imdb._input_path, pid)
                ped_personal_image[sid][vid][pid] = []
                ped_personal_bbox[sid][vid][pid] = []
                num_files = len(os.listdir(input_subpath))
                if num_files < sequence_size:
                    continue
                # if num_files < min_file_num:
                #     print("-------"+pid)
                #     min_file_num = num_files
                index_file = 0
                sequence_index = 0
                for root, dirs, files in os.walk(input_subpath):
                    for file in files:
                        if index_file % (
                                int(math.floor(num_files / sequence_size))) != 0 or sequence_index >= sequence_size:
                            index_file += 1
                            continue
                        else:
                            sequence_index += 1
                            ped_personal_image[sid][vid][pid].append(
                                (torch.from_numpy(cv2.imread(join(input_subpath, file)).astype("f").transpose(2, 0,
                                                                                                              1) / 128.0 - 1.0)))  # .unsqueeze(0)
                            curr_frames = database[sid][vid]['ped_annotations'][pid]['frames']
                            ped_personal_bbox[sid][vid][pid].append(
                                database[sid][vid]['ped_annotations'][pid]['bbox'][curr_frames.index(int(file[:-4]))])
                            # cv2.imwrite('wtf.png', ped_personal_image[sid][vid][pid][0])
                            ped_personal_intention_prob[sid][vid][pid] = \
                                database[sid][vid]['ped_annotations'][pid]['attributes'][
                                    'intention_prob']
                        index_file += 1
    #             print(len(ped_personal_image[sid][vid][pid]))
    # print(min_file_num)
    ped_personal_images_intention = []
    for sid in ped_personal_image:
        for vid in ped_personal_image[sid]:
            for pid in ped_personal_image[sid][vid]:
                ped_personal_images_intention.append(
                    (ped_personal_image[sid][vid][pid], ped_personal_intention_prob[sid][vid][pid]))
    if shuffle:
        random.shuffle(ped_personal_images_intention)
    return ped_personal_images_intention, ped_personal_bbox


def extract_ped_images():
    for sid in sorted(database):
        if sid != "set05":
            continue
        print(sid)
        for vid in sorted(database[sid]):
            print(vid)
            for pid in database[sid][vid]['ped_annotations'].keys():
                print(pid)
                extract_every_ped_image(sid, vid, pid, first_frame=-1, last_frame=-1)


# extract specific pedestrian's serial images
def extract_every_ped_image(sid, vid, pid, first_frame=-1, last_frame=-1, extract_size=None):
    if extract_size is None:
        extract_size = (128, 128)
    curr_frames = database[sid][vid]['ped_annotations'][pid]['frames']
    curr_bboxes = database[sid][vid]['ped_annotations'][pid]['bbox']
    video_path = join(imdb._pie_path, "images", sid, vid)
    if first_frame == -1 and last_frame == -1:
        first_frame = database[sid][vid]['ped_annotations'][pid]['attributes']['exp_start_point']
        last_frame = database[sid][vid]['ped_annotations'][pid]['attributes']['critical_point']
    for frame in range(first_frame, last_frame):
        if frame not in curr_frames:
            continue
        frame_index = curr_frames.index(frame)
        bbox = curr_bboxes[frame_index]
        if frame < 100:
            temp_image = cv2.imread(join(video_path, ("000" + str(frame) + ".png")))
        elif frame < 1000:
            temp_image = cv2.imread(join(video_path, ("00" + str(frame) + ".png")))
        elif frame < 10000:
            temp_image = cv2.imread(join(video_path, ("0" + str(frame) + ".png")))
        else:
            temp_image = cv2.imread(join(video_path, (str(frame) + ".png")))
        # print(bbox)
        up_most = int(round((bbox[3] + bbox[1]) / 2)) - int(round((bbox[3] - bbox[1])))
        down_most = int(round((bbox[3] + bbox[1]) / 2)) + int(round((bbox[3] - bbox[1])))
        left_most = int(round((bbox[0] + bbox[2]) / 2)) - int(round((bbox[3] - bbox[1])))
        right_most = int(round((bbox[0] + bbox[2]) / 2)) + int(round((bbox[3] - bbox[1])))
        # print(up_most,left_most,down_most,right_most)
        loss = [0, 0, 0, 0]
        if left_most < 0:
            loss[0] = -left_most
            left_most = 0
        if right_most > 1600:
            loss[1] = right_most - 1600
            right_most = 1600
        if up_most < 0:
            loss[2] = -left_most
            up_most = 0
        if down_most > 900:
            loss[3] = down_most - 900
            down_most = 900
        temp_image = temp_image[up_most:down_most, left_most:right_most, :]
        temp_image = cv2.copyMakeBorder(temp_image, loss[2], loss[3], loss[0], loss[1], cv2.BORDER_CONSTANT, value=0)
        temp_image = cv2.resize(temp_image, extract_size)
        if not isdir(imdb._input_path):
            makedirs(imdb._input_path)
        input_subpath = join(imdb._input_path, pid)
        if not isdir(input_subpath):
            makedirs(input_subpath)
        # if not isfile(join(video_images_path, "%05.f.png") % frame_num):
        cv2.imwrite(join(input_subpath, f"{frame}.png"), temp_image)

# If we iter every frame and store them into a dataset
# too slow for cv2.imread to read all the images (30-60s for 1 video)
# IMAGE_SET = {}
# for set_id in set_folders:
#     IMAGE_SET[set_id] = {}
#     print('Current Set ID: ', set_id)
#     extract_frames = imdb.get_annotated_frame_numbers(set_id)
#     set_images_path = join(imdb.pie_path, "images", set_id)
#     for vid, frames in sorted(extract_frames.items()):
#         IMAGE_SET[set_id][vid] = {}
#         print(vid)
#         frames_list = frames[1:]
#         video_images_path = join(set_images_path, vid)
#         for frame in frames_list:
#             if frame < 10000:
#                 IMAGE_SET[set_id][vid][frame] = cv2.imread(join(video_images_path, ("0" + str(frame) + ".png")))
#             else:
#                 IMAGE_SET[set_id][vid][frame] = cv2.imread(join(video_images_path, (str(frame) + ".png")))
