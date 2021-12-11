import os
from PIL import Image
import re
import numpy as np
import tensorflow.keras.backend as kb


def create_rotate_img():
    # train satimg
    train_root_dir = "training/images/"
    train_img_file = os.listdir(train_root_dir)
    a = len(train_img_file)

    # train gt
    train_gt_root_dir = "training/groundtruth/"
    train_gt_img_file = os.listdir(train_gt_root_dir)

    image_path = "training/images_rotated"
    image_path_gt = "training/groundtruth_rotated"

    for i in range(a):
        t = Image.open(train_root_dir + train_img_file[i])
        t_gt = Image.open(train_gt_root_dir + train_gt_img_file[i])

        for j in range(0, 360, 90):
            t.rotate(j).save(f"{image_path}/rot{i}_{j}.png")
            t_gt.rotate(j).save(f"{image_path_gt}/rot{i}_{j}.png")


def extract_from_folders():
    for i in range(1, 51):
        path_deep = 'test_set_images/test_' + \
            str(i) + '/test_' + str(i) + '.png'
        path_shallow = 'test_set_images/test_' + str(i) + '.png'
        os.replace(path_deep, path_shallow)


def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def create_submission(filename, Y_test):
    def create_ids_for_submission():
        a = 50
        b = 592
        ids = []
        for x in range(1, a+1):
            for y in range(0, b+1, 16):
                for z in range(0, b+1, 16):
                    ids.append(("00" if len(str(x)) == 1 else "0") +
                               str(x) + "_" + str(y) + "_" + str(z))
        return ids

    ids = create_ids_for_submission()

    dir = 'submissions/'
    f = open(dir + filename, 'a')
    f.write("id,prediction\n")
    for i in range(len(Y_test)):
        f.write(str(ids[i]) + "," + str(Y_test[i]) + "\n")
    f.close()


def to_numpy(images, width, height):
    arr = []
    for i in images:
        whole_img = np.asarray(Image.open(i))
        for w in range(0, whole_img.shape[1], width):
            for h in range(0, whole_img.shape[0], height):
                if whole_img.ndim == 3:
                    arr.append(whole_img[w:w+width, h:h+height, :])
                else:
                    arr.append(whole_img[w:w+width, h:h+height])
    return np.array(arr)


def get_random_train(imgs, gt, crop_width, crop_height):
    x = []
    y = []
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    range_w = width - crop_width
    range_h = height - crop_height
    for i in range(len(imgs)):
        start_w = np.random.randint(0, range_w + 1)
        start_h = np.random.randint(0, range_h + 1)
        x.append(imgs[i][start_w:start_w+crop_width,
                 start_h:start_h+crop_height, :])
        y.append(gt[i][start_w:start_w+crop_width,
                 start_h:start_h+crop_height, :])
    return np.asarray(x), np.asarray(y)


def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def dice_coef(y_true, y_pred, smooth=1):
    intersection = kb.sum(kb.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (kb.sum(kb.square(y_true), -1) + kb.sum(kb.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
