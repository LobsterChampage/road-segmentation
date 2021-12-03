#!/usr/bin/python
from PIL import Image
import math
import numpy as np

def mask(submission):
    w, h = 16, 16
    imgwidth = int(math.ceil((600.0/w))*w)
    imgheight = int(math.ceil((600.0/h))*h)

    for nr in range(1, 51):
        im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
        f = open(submission)
        lines = f.readlines()
        image_id_str = '%03d_' % nr
        for i in range(1, len(lines)):
            line = lines[i]
            if not image_id_str in line:
                continue

            tokens = line.split(',')
            id = tokens[0]
            prediction = int(tokens[1])
            tokens = id.split('_')
            i = int(tokens[1])
            j = int(tokens[2])

            je = min(j+w, imgwidth)
            ie = min(i+h, imgheight)
            if prediction == 0:
                adata = np.zeros((w,h))
            else:
                adata = np.ones((w,h))

            im[j:je, i:ie] = (adata * 255).round().astype(np.uint8)

        Image.fromarray(im).save('predictions/prediction_' + '%03d' % nr + '.png')