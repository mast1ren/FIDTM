import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''
shanghai_root = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech'
jhu_root = '/home/dkliang/projects/synchronous/dataset/jhu_crowd_v2.0'
qnrf_root = '/home/dkliang/projects/synchronous/dataset/UCF-QNRF_ECCV18'
dronebird_root = './preprocessed_data'
try:
    dronebird_train_path = os.path.join(dronebird_root, 'train/images')
    dronebird_test_path = os.path.join(dronebird_root, 'test/images')
    dronebird_val_path = os.path.join(dronebird_root, 'val/images')
    train_list = []
    for file in os.listdir(dronebird_train_path):
        if file.endswith('.jpg'):
            train_list.append(os.path.join(dronebird_train_path, file))
    train_list.sort()
    np.save('./npydata/dronebird_train.npy', train_list)
    test_list = []
    for file in os.listdir(dronebird_test_path):
        if file.endswith('.jpg'):
            test_list.append(os.path.join(dronebird_test_path, file))
    test_list.sort()
    np.save('./npydata/dronebird_test.npy', test_list)
    val_list = []
    for file in os.listdir(dronebird_val_path):
        if file.endswith('.jpg'):
            val_list.append(os.path.join(dronebird_val_path, file))
    val_list.sort()
    np.save('./npydata/dronebird_val.npy', val_list)
    print('dronebird dataset done')
except:
    print('dronebird dataset error')

try:

    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiBtrain_path = shanghai_root + '/part_B_final/train_data/images/'
    shanghaiBtest_path = shanghai_root + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully")
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = qnrf_root + '/train_data/images/'
    Qnrf_test_path = qnrf_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)
    print("Generate QNRF image list successfully")
except:
    print("The QNRF dataset path is wrong. Please check your path.")

try:

    Jhu_train_path = jhu_root + '/train/images_2048/'
    Jhu_val_path = jhu_root + '/val/images_2048/'
    jhu_test_path = jhu_root + '/test/images_2048/'

    train_list = []
    for filename in os.listdir(Jhu_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Jhu_train_path + filename)
    train_list.sort()
    np.save('./npydata/jhu_train.npy', train_list)

    val_list = []
    for filename in os.listdir(Jhu_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(Jhu_val_path + filename)
    val_list.sort()
    np.save('./npydata/jhu_val.npy', val_list)

    test_list = []
    for filename in os.listdir(jhu_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(jhu_test_path + filename)
    test_list.sort()
    np.save('./npydata/jhu_test.npy', test_list)

    print("Generate JHU image list successfully")
except:
    print("The JHU dataset path is wrong. Please check your path.")

try:
    f = open("./data/NWPU_list/train.txt", "r")
    train_list = f.readlines()

    f = open("./data/NWPU_list/val.txt", "r")
    val_list = f.readlines()

    f = open("./data/NWPU_list/test.txt", "r")
    test_list = f.readlines()

    root = '/home/dkliang/projects/synchronous/dataset/NWPU_localization/images_2048/'
    train_img_list = []
    for i in range(len(train_list)):
        fname = train_list[i].split(' ')[0] + '.jpg'
        train_img_list.append(root + fname)
    np.save('./npydata/nwpu_train_2048.npy', train_img_list)

    val_img_list = []
    for i in range(len(val_list)):
        fname = val_list[i].split(' ')[0] + '.jpg'
        val_img_list.append(root + fname)
    np.save('./npydata/nwpu_val_2048.npy', val_img_list)

    test_img_list = []
    root = root.replace('images', 'test_data')
    for i in range(len(test_list)):
        fname = test_list[i].split(' ')[0] + '.jpg'
        fname = fname.split('\n')[0] + fname.split('\n')[1]
        test_img_list.append(root + fname)

    np.save('./npydata/nwpu_test_2048.npy', test_img_list)
    print("Generate NWPU image list successfully")
except:
    print("The NWPU dataset path is wrong. Please check your path.")
