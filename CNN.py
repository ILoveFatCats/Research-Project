import numpy as np
import pandas as pd 
import os
import tensorflow
import keras.api._v2.keras as keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten,Conv2D,MaxPool2D,Dense,ZeroPadding2D
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import random
from PIL import Image

path = "E:/Aplikacje/Python 3.9/Projects/Research Project/train"
category = []
count = []
img_path = []
class_ = []
for folder in os.listdir(path):
    i = 0
    category.append(folder)
    for subfolder in os.listdir(os.path.join(path,folder)):
        i += 1
        img_path.append(os.path.join(path,folder,subfolder))
        class_.append(folder)
    count.append(i)
df = pd.DataFrame()
df["specie_category"] = category
df["count"] = count

sns.countplot(x = df["specie_category"])
plt.plot()

plt.figure(1, figsize=(25, 25))

n = 0
for i in range(16):
    j = random.randint(0,2358)
    n += 1
    plt.subplot(4, 4, n)
    random_img = img_path[j]
    imgs = cv2.imread(random_img)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imgs, cmap=plt.cm.binary)
    plt.xlabel(class_[j])
plt.show()

train_dir = 'E:/Aplikacje/Python 3.9/Projects/Research Project/train'
#Object of ImageDataGenrator, can specify data augmentation steps here
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir,batch_size=32,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
test_dir = 'E:/Aplikacje/Python 3.9/Projects/Research Project/test'
#Object of ImageDataGenrator, can specify data augmentation steps here
test_datagen = ImageDataGenerator(rescale = 1.0/255,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')
test_generator = test_datagen.flow_from_directory(test_dir,batch_size=32,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))

model = Sequential([
    Conv2D(32,3,activation='relu',input_shape=(150, 150,3)),
    MaxPool2D(pool_size=2,strides=2),
    ZeroPadding2D(padding=(2,2)),
    Conv2D(64,3,activation='relu'),
    MaxPool2D(pool_size=2,strides=2),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(400,activation="softmax")
])

model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(train_generator,validation_data = test_generator,epochs=10,batch_size=256)

from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras import layers
from keras import Model
local_weights_file = '../input/inception-weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
#pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(400, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

history=model.fit(train_generator,validation_data = test_generator,epochs=10,batch_size=256)

from keras.applications import VGG16

pretrained_model=VGG16(input_shape = (150, 150, 3), 
                        include_top = False, 
                        weights = 'imagenet')

for layer in pretrained_model.layers:
     layer.trainable = False

# pretrained_model.summary()
last_layer = pretrained_model.get_layer('block5_pool')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output= last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(400, activation='softmax')(x)           

model_vgg = Model(pretrained_model.input, x) 


model_vgg.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

model_vgg.fit(train_generator,validation_data = test_generator,epochs=10,batch_size=256)