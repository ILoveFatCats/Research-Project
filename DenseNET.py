import subprocess
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL

# First connect to firewall (access node) then to the workstation!
# TO CONNECT TO ACCESS NODE (local terminal):   "gk99@GK99:~$ ssh s175405@kask.eti.pg.gda.pl"       password: same as steam password
# TO CONNECT TO SANNA.KASK (access termianl):   "gk99@GK99:~$ ssh s175405@sanna.kask"               password: same as steam password
# TO UPDATE THE CODE:                           "gk99@GK99:~/tmp/ResearchProject$ scp DenseNET.py s175405@kask.eti.pg.gda.pl:ResearchProject12KASK/"
# !!! DISCLAIMER: RUN ABOVE COMMAND             
#     FROM THE LOCAL TERMINAL !!!
# TO RUN THE APPLICATION:                       "s175405@sanna:~$ python3 /home/macierz/s175405/ResearchProject12KASK/DenseNET.py"
# TO CREATE THE POWER COMSUMPTION LOG:          "s175405@sanna:~/ResearchProject12KASK$ nvidia-smi dmon -s p -o T -f power1.txt"
#                                               !REMEMBER TO SPECIFY THE GPU/s ID/IDs, OTHERWISE YOU'LL GET THE LOGS OF IDLE GPUs!
# !!! DO-TO: explain what the individual parameters are responsible for !!!
# TO DOWNLOAD THE LOGS FORM SERVER:             "gk99@GK99:~$ scp -p s175405@kask.eti.pg.gda.pl:ResearchProject12KASK/power1.txt ~/Apps/"
#                                               "gk99@GK99:~$ scp -p s175405@kask.eti.pg.gda.pl:ResearchProject12KASK/yoko.csv ~/Apps/"

#   nvidia-smi cheat sheet:                     https://www.seimaxim.com/kb/gpu/nvidia-smi-cheat-sheet
#   You can confirm that this is happening by using nvidia-smi to monitor the GPUs while your application is running.
#   nvidia-smi dmon 
#   Determine the current, default and maximum power limit as follows:
#   nvidia-smi -q | grep 'Power Limit'
#   Ensure that persistence mode is being used.
#   Increase the SW Power Cap limit for all GPUs as follows, where xxx is the desired value in watts:
#   nvidia-smi -pl xxx
#   (example):    sudo nvidia-smi --id=0 -pl 200
#   NVIDIA Quadro RTX 5000 power level range: 125W - 230W
#   NVIDIA Quadro RTX 6000 power level range: 100W - 260W


#   temp:
#   yokotool /dev/usbtmc0 read T,P -o yoko.csv
#   yokotool /dev/usbtmc0 --pmtype wt310 --baudrate=9600 info
#
#

#import wandb
#wandb.init()
#wtviewer

def running_commands():
    nvsmi = subprocess.run('/home/macierz/s175405/ResearchProject12KASK/nvsmiPowerLog.sh', shell=True, capture_output=False)
    yokotool = subprocess.run('/home/macierz/s175405/ResearchProject12KASK/yokotoolPowerLog.sh', shell=True, capture_output=False)
running_commands()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setting up the distributed training

# Here You can choose the used GPUs:
GPUs = ["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5", "GPU:6", "GPU:7"]
strategy = tf.distribute.MirroredStrategy(GPUs)                     # Distribution of training
BATCH_SIZE = 512                                                    # Default value is: 128 (it yields the best performance)
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync      # Auto-scalability of our training

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

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

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=GLOBAL_BATCH_SIZE,
    image_size=(224,224),
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    directory=valid_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=GLOBAL_BATCH_SIZE,
    image_size=(224,224),
)

class_names = train_ds.class_names

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    print(len(images))
    for i in range(2):
        ax = plt.subplot(1,2, i + 1)
        plt.imshow( images[i].numpy().astype('uint8') )
        plt.axis('off')

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

#===
def get_model():
    densenet_raw_model = tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3), # input size of image fixed
        pooling='max',
    )
    densenet_raw_model.trainable = False

    model = keras.Sequential([  
        densenet_raw_model,
        layers.Flatten(),
        layers.Dense(units=2000,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=400, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === FOR MULTI GPU TRAINING
with strategy.scope():
    gpu_model = get_model()
gpu_model.fit(train_ds, epochs = 3)

# === FOR SINGLE GPU TRAINING
#gpu_model = get_model()
#gpu_model.fit(train_ds, epochs = 3)

exit()
