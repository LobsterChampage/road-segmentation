import os
from PIL import Image
import re

def create_rotate_img():
    #train satimg
    train_root_dir = "training/images/"
    train_img_file = os.listdir(train_root_dir)
    a = len(train_img_file)

    #train gt
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
    for i in range(1,51):
        path_deep = 'test_set_images/test_' + str(i) + '/test_' + str(i) + '.png'
        path_shallow = 'test_set_images/test_' + str(i) + '.png'
        os.replace(path_deep, path_shallow)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def create_submission(filename, Y_test):
    def create_ids_for_submission():
        a = 50
        b = 592
        ids = []
        for x in range(1, a+1):
            for y in range(0, b+1, 16):
                for z in range(0, b+1, 16):
                    ids.append(("00" if len(str(x)) == 1 else "0") + str(x) + "_" + str(y) + "_" + str(z))
        return ids
    
    ids = create_ids_for_submission()
    
    dir = 'submissions/'
    f = open(dir + filename, 'a')
    f.write("id,prediction\n")
    for i in range(len(Y_test)):
        f.write(str(ids[i]) + "," + str(Y_test[i]) + "\n")
    f.close()