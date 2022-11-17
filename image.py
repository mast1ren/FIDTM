import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2
import os


def load_data_fidt(img_path, args, train=True):
    # img_path = os.path.join('./preprocessed_data', img_path)
    gt_path = os.path.join(os.path.dirname(img_path).replace('images', 'gt_fidt_map'), 'GT_' + os.path.basename(img_path).replace('.jpg', '.h5'))
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
