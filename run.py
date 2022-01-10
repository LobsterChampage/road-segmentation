from matplotlib import pyplot as plt
import tensorflow as tf
from functions import *
import os
import math
import glob
import numpy as np

import tensorflow_addons as tfa

#CONFIG
crops = 3
rots = 4
split = 0.97
BATCH_SIZE = 4
#config aftertrain
thresh = 0.26

if __name__ == "__main__":
    imgs = tf.data.Dataset.list_files('training/images/*')

    train_size = int(len(imgs)*split)
    train_ds = imgs.take(train_size) #train-test split happens here
    test_ds = imgs.skip(train_size)

    def process_image(file_path):
        img = tf.io.read_file(file_path) # load the raw data from the file as a string
        img = tf.io.decode_image(img)/255 #to have numbers in 0-1 range
        
        parts = tf.strings.split(file_path, os.path.sep)
        tr = tf.constant('groundtruth')
        path = tf.strings.join((parts[0], tr, parts[2]), separator='\\')
        gt = tf.io.read_file(path) # load the raw data from the file as a string
        gt = tf.io.decode_image(gt)/255
        
        img1 = tf.image.rot90(img, k=0, name=None)
        gt1 = tf.image.rot90(gt, k=0, name=None)
        img2 = tfa.image.rotate(img, math.radians(45), interpolation='BILINEAR')
        gt2 = tfa.image.rotate(gt, math.radians(45), interpolation='BILINEAR')

        dataset = tf.data.Dataset.from_tensors((img1, gt1))
        dataset = dataset.concatenate(tf.data.Dataset.from_tensors((img2, gt2)))
        
        for k in range(1,rots):
            img3 = tf.image.rot90(img, k=k, name=None) #rotations for each picture
            gt3 = tf.image.rot90(gt, k=k, name=None)
            img4 = tfa.image.rotate(img3, math.radians(45), interpolation='BILINEAR')
            gt4 = tfa.image.rotate(gt3, math.radians(45), interpolation='BILINEAR')

            dataset = dataset.concatenate(tf.data.Dataset.from_tensors((img3, gt3)))
            dataset = dataset.concatenate(tf.data.Dataset.from_tensors((img4, gt4)))
        return dataset

    def crop(img, gt):
        ite = math.floor(96/crops)
        
        seed=(int(np.random.rand(1)[0]*10000000),int(np.random.rand(1)[0]*10000000)) #to have GT and satImg be same crop
        base_img = tf.image.stateless_random_crop(img, (304,304,3), seed) #1 random crop outside of loop
        base_gt = tf.image.stateless_random_crop(gt, (304,304,1), seed)
        dataset = tf.data.Dataset.from_tensors((base_img, base_gt))
        
        for i in range(0, 96, ite):
            for j in range(0, 96, ite):#what we call grid crop
                img1 = tf.image.crop_to_bounding_box(img, i, j, 304, 304)
                gt1 = tf.image.crop_to_bounding_box(gt, i, j, 304, 304)
                img2 = tf.image.flip_left_right(img1)
                gt2 = tf.image.flip_left_right(gt1)

                dataset = dataset.concatenate(tf.data.Dataset.from_tensors((img1, gt1)))
                dataset = dataset.concatenate(tf.data.Dataset.from_tensors((img2, gt2)))
        return dataset

    train_ds = train_ds.flat_map(process_image)
    train_ds = train_ds.shuffle(5000)
    train_ds = train_ds.cache()
    train_ds = train_ds.flat_map(crop)
    train_ds = train_ds.shuffle(10000) #shuffle after processing to ensure distributed pictures
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.flat_map(process_image)
    test_ds = test_ds.flat_map(crop)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    #Build the model
    inputs = tf.keras.layers.Input((304, 304, 3))

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(0.2), kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(0.2), kernel_initializer='he_normal', padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(0.1), kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) #early stop
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=dice_coef_loss,
                metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #here epochs can be changed
    history = model.fit(train_ds, batch_size=4, epochs=100, callbacks=[callback])
    model.evaluate(test_ds)


    test_images = glob.glob('test_set_images/*') #load aicrowd test-set
    test_images = sorted_alphanumeric(test_images) #sort in case it loads in weird order
    np_test_imgs = to_numpy(test_images, 304, 304)
    np_test_imgs = np_test_imgs.astype(np.float32)/256 #range 0-1 in numpy of image
    pr = model.predict(np_test_imgs)

    def outputx16(nump, treshhold): #convert from pixelwise prediction to patch prediction
        pred = []
        for h in range(0,nump.shape[1],16):
            for w in range(0,nump.shape[0],16):
                if nump[w:w+16,h:h+16,:].sum()/238 > treshhold:
                    pred.append(1)
                else:
                    pred.append(0)
        return np.asarray(pred)

    # the two following loops is to reconstruct aicrowd test-set from 4 images of 304x304
    sized_pr = []
    for i in range(0,int(len(pr)),4):
        a = np.concatenate((pr[i],pr[i+1]),axis=1)
        b = np.concatenate((pr[i+2],pr[i+3]),axis=1)
        sized_pr.append(np.concatenate((a,b),axis=0))
    sized_pr = np.asarray(sized_pr)

    sized_img = []
    for i in range(0,int(len(np_test_imgs)),4):
        a = np.concatenate((np_test_imgs[i],np_test_imgs[i+1]),axis=1)
        b = np.concatenate((np_test_imgs[i+2],np_test_imgs[i+3]),axis=1)
        sized_img.append(np.concatenate((a,b),axis=0))
    sized_img = np.asarray(sized_img)

    #this is to show prediction beside satimg
    for i in range(50):
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(1, 3, 1)
        plt.imshow(sized_img[i], interpolation='nearest')
        plt.subplot(1, 3, 2)
        plt.imshow(np.reshape(sized_pr[i],(608,608)), interpolation='nearest')
        plt.subplot(1, 3, 3)
        c = np.reshape(outputx16(sized_pr[i], thresh),(38,38)).T
        plt.imshow(c, interpolation='nearest')
        plt.show()

    a = outputx16(sized_pr[0], thresh)
    for i in range(1, len(sized_pr)):
        b = outputx16(sized_pr[i], thresh)
        a = np.concatenate((a,b),axis=0)

    #creates a list of ids by the submission format
    ids = []
    for i in range(1,51):
        for j in range(0,593, 16):
            for k in range(0,593, 16):
                ids.append('{:03}_{}_{}'.format(i,j,k))
    ids = np.array(ids)

    a = a.astype(str)

    dirr = 'submissions/sub.csv'
    #creates submission
    f = open(dirr, "w")
    f.write('id,prediction\n')

    for i in range(a.shape[0]):
        f.write(ids[i] + ',' + a[i] + '\n')
    f.close()
    #puts masks in /predictions
    from submission_to_mask import mask

    mask(dirr)