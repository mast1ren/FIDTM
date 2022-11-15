import os
import time
import json

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import torch

'''please set the dataset path'''
root = '../../../ds/dronebird'

img_train_path = os.path.join(root, 'train')
gt_train_path = os.path.join(root, 'train')
img_test_path = os.path.join(root, 'test')
gt_test_path = os.path.join(root, 'test')

save_train_img_path = '../preprocessed_data/train/images/'
save_train_gt_path = '../preprocessed_data/train/gt_fidt_map/'
save_val_img_path = '../preprocessed_data/val/images/'
save_val_gt_path = '../preprocessed_data/val/gt_fidt_map/'
save_test_img_path = '../preprocessed_data/test/images/'
save_test_gt_path = '../preprocessed_data/test/gt_fidt_map/'

if not os.path.exists(save_train_img_path):
    os.makedirs(save_train_img_path)

if not os.path.exists(save_train_gt_path):
    os.makedirs(save_train_gt_path)

# if not os.path.exists(save_train_img_path.replace('images', 'gt_show_fidt')):
#     os.makedirs(save_train_img_path.replace('images', 'gt_show_fidt'))

if not os.path.exists(save_val_img_path):
    os.makedirs(save_val_img_path)

if not os.path.exists(save_val_gt_path):
    os.makedirs(save_val_gt_path)

if not os.path.exists(save_test_img_path):
    os.makedirs(save_test_img_path)

if not os.path.exists(save_test_gt_path):
    os.makedirs(save_test_gt_path)

# if not os.path.exists(save_test_img_path.replace('images', 'gt_show_fidt')):
#     os.makedirs(save_test_img_path.replace('images', 'gt_show_fidt'))

distance = 1
img_train = []
gt_train = []
img_val = []
gt_val = []
img_test = []
gt_test = []

with open(os.path.join(root, 'train.json'), 'r') as f:
    data = json.load(f)

for i in range(len(data)):
    img_train.append(os.path.join(root, data[i]))
    gt_train.append(os.path.join(root, os.path.dirname(data[i]).replace('images', 'ground_truth'), 'GT_' + os.path.basename(data[i]).replace('.jpg', '.mat')))

with open(os.path.join(root, 'val.json'), 'r') as f:
    data = json.load(f)

for i in range(len(data)):
    img_test.append(os.path.join(root, data[i]))
    gt_test.append(os.path.join(root, os.path.dirname(data[i]).replace('images', 'ground_truth'), 'GT_' + os.path.basename(data[i]).replace('.jpg', '.mat')))

with open(os.path.join(root, 'test.json'), 'r') as f:
    data = json.load(f)

for i in range(len(data)):
    img_test.append(os.path.join(root, data[i]))
    gt_test.append(os.path.join(root, os.path.dirname(data[i]).replace('images', 'ground_truth'), 'GT_' + os.path.basename(data[i]).replace('.jpg', '.mat')))

img_train.sort()
gt_train.sort()
img_test.sort()
gt_test.sort()
# print(img_train)
# print(gt_train)
print(len(img_train), len(gt_train), len(img_test), len(gt_test))


''''generate fidt map'''


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map,
                        0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map


'''for training dataset'''
for k in range(len(img_train)):

    Img_data = cv2.imread(img_train[k])
    Gt_data = scipy.io.loadmat(gt_train[k])
    rate = 1
    rate_1 = 1
    rate_2 = 1
    flag = 0
    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[1]
        flag = 1
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[0]
        flag = 1
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_1)

    min_shape = 512.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate_2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate_2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_2)

    rate = rate_1 * rate_2

    Gt_data = Gt_data['locations']
    Gt_data = Gt_data * rate
    fidt_map = fidt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    new_img_path = os.path.join(save_train_img_path + os.path.basename(img_train[k]))

    # mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images', 'gt_show_fidt')
    h5_path = os.path.join(save_test_gt_path, os.path.basename(gt_test[k]).replace('.mat', '.h5'))

    print(img_train[k], np.sum(kpoint))

    kpoint = kpoint.astype(np.uint8)
    with h5py.File(h5_path, 'w') as hf:
        hf['kpoint'] = kpoint
        hf['fidt_map'] = fidt_map

    cv2.imwrite(new_img_path, Img_data)

#     fidt_map = fidt_map
#     fidt_map = fidt_map / np.max(fidt_map) * 255
#     fidt_map = fidt_map.astype(np.uint8)
#     fidt_map = cv2.applyColorMap(fidt_map, 2)

#     result = fidt_map

#     cv2.imwrite(gt_show_path, result)


'''for val dataset'''
for k in range(len(img_val)):
    Img_data = cv2.imread(img_val[k])
    Gt_data = scipy.io.loadmat(gt_val[k])

    rate_1 = 1
    rate_2 = 1

    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[1]
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_1)

    min_shape = 1024.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate_2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate_2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_2)

    rate = rate_1 * rate_2

    print(img_test[k], Img_data.shape)

    Gt_data = Gt_data['locations']
    Gt_data = Gt_data * rate

    fidt_map = fidt_generate1(Img_data, Gt_data, 1)
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] += 1

    new_img_path = os.path.join(save_val_img_path, os.path.basename(img_val[k]))

    # mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images', 'gt_show_fidt')
    h5_path = os.path.join(save_val_gt_path, os.path.basename(gt_val[k]).replace('.mat', '.h5'))

    kpoint = kpoint.astype(np.uint8)
    with h5py.File(h5_path, 'w') as hf:
        hf['kpoint'] = kpoint
        hf['fidt_map'] = fidt_map

    cv2.imwrite(new_img_path, Img_data)


'''for testing dataset'''
for k in range(len(img_test)):
    Img_data = cv2.imread(img_test[k])
    Gt_data = scipy.io.loadmat(gt_test[k])

    rate_1 = 1
    rate_2 = 1

    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[1]
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_1)

    min_shape = 1024.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate_2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate_2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_2)

    rate = rate_1 * rate_2

    print(img_test[k], Img_data.shape)

    Gt_data = Gt_data['locations']
    Gt_data = Gt_data * rate

    fidt_map = fidt_generate1(Img_data, Gt_data, 1)
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] += 1

    new_img_path = os.path.join(save_test_img_path, os.path.basename(img_test[k]))

    # mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images', 'gt_show_fidt')
    h5_path = os.path.join(save_test_gt_path, os.path.basename(gt_test[k]).replace('.mat', '.h5'))

    kpoint = kpoint.astype(np.uint8)
    with h5py.File(h5_path, 'w') as hf:
        hf['kpoint'] = kpoint
        hf['fidt_map'] = fidt_map

    cv2.imwrite(new_img_path, Img_data)

    # fidt_map = fidt_map
    # fidt_map = fidt_map / np.max(fidt_map) * 255
    # fidt_map = fidt_map.astype(np.uint8)
    # fidt_map = cv2.applyColorMap(fidt_map, 2)

    # # result = np.hstack((mask_map,fidt_map))
    # result = fidt_map

    # cv2.imwrite(gt_show_path, result)
