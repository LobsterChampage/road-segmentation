import os
from PIL import Image

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