import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# First connect to firewall (access node) then to the workstation!
# TO CONNECT TO ACCESS NODE:                "ssh s175405@kask.eti.pg.gda.pl"     password: same as steam password
# TO CONNECT TO SANNA.KASK:                 "ssh s175405@sanna.kask"             password: same as steam password
# TO UPDATE THE CODE:                       "scp DenseNET.py s175405@kask.eti.pg.gda.pl:ResearchProject12KASK/"
# TO DOWNLOAD THE LOGS FORM SERVER:         "scp -p s175405@kask.eti.pg.gda.pl:ResearchProject12KASK/PowerUsageTestLog.txt ~/Apps/
# TO RUN THE APPLICATION:                   "python3 /home/macierz/s175405/ResearchProject12KASK/DenseNET.py"
# To CREATE THE POWER COMSUMPTION LOG:      "nvidia-smi dmon -s p -o T -f <filename.txt>"   
#                                           !REMEMBER TO SPECIFY THE GPU/s ID/IDs, OTHERWISE YOU'LL GET THE LOGS OF IDLE GPUs!


#   You can confirm that this is happening by using nvidia-smi to monitor the GPUs while your application is running.
#   nvidia-smi dmon 
#   Determine the current, default and maximum power limit as follows:
#   nvidia-smi -q | grep 'Power Limit'
#   Ensure that persistence mode is being used.
#   Increase the SW Power Cap limit for all GPUs as follows, where xxx is the desired value in watts:
#   nvidia-smi -pl xxx
#   NVIDIA Quadro RTX 5000 power level range: 125W - 230W
#   NVIDIA Quadro RTX 6000 power level range: 100W - 260W

# tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

#print(c)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.listdir('E:/Aplikacje/Python 3.9/Research Project')                 #Windows10      PC directory        GTX 1080ti
#os.listdir('/home/gk99/Apps/python_env/venv/birdsDataset')             #Ubuntu 20.04   laptop directory    RTX 3050 Mobile
os.listdir('/home/macierz/s175405/ResearchProject12KASK/birdsDataset')  #Ubuntu 20.04   cluster directory   Quadro RTX 5000 / Quadro RTX 6000

#data_ = pd.read_csv('E:/Aplikacje/Python 3.9/Research Project/birds.csv')
#data_ = pd.read_csv('/home/gk99/Apps/python_env/venv/birdsDataset/birds.csv')
data_ = pd.read_csv('/home/macierz/s175405/ResearchProject12KASK/birdsDataset/birds.csv')
data_.head()

#dir = 'E:/Aplikacje/Python 3.9/Research Project/'
#dir = '/home/gk99/Apps/python_env/venv/birdsDataset/'
dir = '/home/macierz/s175405/ResearchProject12KASK/birdsDataset/'
train_dir = dir + 'train'
valid_dir = dir + 'test'

# total number of species
species_count = len(data_['class index'].unique())

train_ds = keras.utils.image_dataset_from_directory(directory=train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=128,
    image_size=(224,224),
)

valid_ds = keras.utils.image_dataset_from_directory(
    directory=valid_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=128,
    image_size=(224,224),
)

class_names = train_ds.class_names

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    print(len(images))
    for i in range(2):
        ax = plt.subplot(1,2, i + 1)
        plt.imshow( images[i].numpy().astype('uint8') )
#         plt.title(class_names[labels[i]])
        plt.axis('off')

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

densenet_raw_model = tf.keras.applications.densenet.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3), # input size of image fixed
    pooling='max',
)

# to not train thse weights with our training images
for layers in densenet_raw_model.layers:
    layers.trainable=False
    
# apped Dense layers to classify it for the number of classes for the given problem
num_classes = len(class_names)
densenet_model_output = densenet_raw_model.layers[-1].output
appended_layer = tf.keras.layers.Flatten()(densenet_model_output)

appended_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(appended_layer)

# final model

densenet_final_model = tf.keras.Model(densenet_raw_model.input, appended_layer)

#print(densenet_final_model.summary())
densenet_final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = densenet_final_model.fit(
    train_ds,
    epochs=5,
    validation_data=valid_ds
)
