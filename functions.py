import os
from PIL import Image
import re
import numpy as np
import tensorflow.keras.backend as kb


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
