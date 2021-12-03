import os,sys
from PIL import Image

def create_rotate_img():
    #train normal
    train_root_dir = "training/images/"
    folders_train = os.listdir(train_root_dir)
    a = len(folders_train)
    train_img_file = os.listdir(train_root_dir)
    print("Loading training images: " + str(a))

    #train gt
    train_gt_root_dir = "training/groundtruth/"
    folders_gt_train = os.listdir(train_gt_root_dir)
    b = len(folders_gt_train)
    train_gt_img_file = os.listdir(train_gt_root_dir)
    print("Loading ground truth training images: " + str(b))
    
    t = Image.open(train_root_dir + train_img_file[0])
    t_gt = Image.open(train_gt_root_dir + train_gt_img_file[0])

    image_path = "training/images_rotated"
    image_path_gt = "training/groundtruth_rotated"

    #os.mkdir(image_path)
    #os.mkdir(image_path_gt)

    for i in range(a):
        t = Image.open(train_root_dir + train_img_file[i])
        t_gt = Image.open(train_gt_root_dir + train_gt_img_file[i])

        for j in range(0, 360, 90):
            rot_t = t.transpose(Image.rotate(j))
            rot_t_gt = t_gt.transpose(Image.rotate(j))

            rot_t = rot_t.save(f"{image_path}/rot{i}_{j}.png")
            rot_t_gt = rot_t_gt.save(f"{image_path_gt}/rot{i}_{j}.png")