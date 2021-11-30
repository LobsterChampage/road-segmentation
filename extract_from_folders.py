import os

for i in range(1,51):
    path_deep = 'test_set_images/test_' + str(i) + '/test_' + str(i) + '.png'
    path_shallow = 'test_set_images/test_' + str(i) + '.png'
    os.replace(path_deep, path_shallow)