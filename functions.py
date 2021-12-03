def create_rotate_img():
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
            if j == 0:
                rot_t = t.transpose(Image.ROTATE_90)
                rot_t_gt = t_gt.transpose(Image.ROTATE_90)
            elif j == 90:
                rot_t = t.transpose(Image.ROTATE_90)
                rot_t_gt = t_gt.transpose(Image.ROTATE_90)
            elif j == 180:
                rot_t = t.transpose(Image.ROTATE_180)
                rot_t_gt = t_gt.transpose(Image.ROTATE_180)
            else:
                rot_t = t.transpose(Image.ROTATE_270)
                rot_t_gt = t_gt.transpose(Image.ROTATE_180)

            rot_t = rot_t.save(f"{image_path}/rot{i}_{j}.png")
            rot_t_gt = rot_t_gt.save(f"{image_path_gt}/rot{i}_{j}.png")