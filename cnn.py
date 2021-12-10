import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
#from tensorflow import layers
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#train_images =
#train_labels =
#test_images =
"""
X_train = X
y_train = Y
X_test = X_test
"""

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.shape)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#class_names = ['road', 'not_road']

#import sys; sys.exit()

shapeForInput = int(400/16)

#model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32, 32, 3))) # 32x32 RGB images
#model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(shapeForInput, shapeForInput, 3)))
#padding = "same"
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

#import sys; sys.exit()


loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss) #, metrics=metrics)

#training
batch_size = 64
epochs = 5 #affect training

model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
#model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)