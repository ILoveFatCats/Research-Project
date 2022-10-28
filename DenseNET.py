import os
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.xception import Xception

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=500):
    assert (train_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def running_commands():
	nvsmi = subprocess.run('/home/macierz/s175405/ResearchProject12KASK/nvsmiPowerLog.sh', shell=True , capture_output=False)
	yokotool = subprocess.run('/home/macierz/s175405/ResearchProject12KASK/yokotoolPowerLog.sh', shell=True, capture_output=False)
running_commands()

# Setting up the distributed training
# Here You can choose the used GPUs:
GPUs = ["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5", "GPU:6", "GPU:7"]
strategy = tf.distribute.MirroredStrategy(GPUs)                     # Distribution of training
BATCH_SIZE = 512                                                    # Default value is: 512 (it yields the best performance)
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync      # Auto-scalability of our training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.listdir('/home/macierz/s175405/ResearchProject12KASK/birdsDataset')  #Ubuntu 20.04   cluster directory   Quadro RTX 5000 / Quadro RTX 6000
data_ = pd.read_csv('/home/macierz/s175405/ResearchProject12KASK/birdsDataset/birds.csv')
data_.head()
dir = '/home/macierz/s175405/ResearchProject12KASK/birdsDataset/'
train_dir = dir + "train/"
test_dir = dir + "test/"
val_dir = dir + "valid/"

# total number of species
species_count = len(data_['class index'].unique())

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory( train_dir , target_size=(224,224,3) , batch_size=GLOBAL_BATCH_SIZE , class_mode = "categorical" ,shuffle=True )

val_data = val_gen.flow_from_directory( val_dir , target_size=(224,224,3) , batch_size=GLOBAL_BATCH_SIZE , class_mode = "categorical" , shuffle=True )

test_data = test_gen.flow_from_directory( test_dir , target_size=(224,224,3) , batch_size=GLOBAL_BATCH_SIZE , class_mode = "categorical" ,shuffle=False )

#=== DEFINING THE MODEL
def get_model():
    xceptionnet = Xception( include_top=False , weights="imagenet" , input_shape=(224,224,3))

    xceptionnet.trainable = False

    model = Sequential([
        xceptionnet,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(400,activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === FOR MULTI GPU TRAINING
with strategy.scope():
    gpu_model = get_model()
history = gpu_model.fit(train_data, epochs = 10, validation_data = val_data)
results = gpu_model.evaluate(test_data)

# === FOR SINGLE GPU TRAINING
#gpu_model = get_model()
#history = gpu_model.fit(train_data, epochs = 10, validation_data = val_data)
#results = gpu_model.evaluate(test_data)

Omae_Wa_Mou_Shindeiru = subprocess.run('/home/macierz/s175405/ResearchProject12KASK/kill.sh', shell=True, capture_output=False)

exit()
