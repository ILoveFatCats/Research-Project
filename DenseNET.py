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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setting up the distributed training

# Here You can choose the used GPUs:
GPUs = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tf.distribute.MirroredStrategy(GPUs)
BATCH_SIZE = 1024                                                  
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.listdir('/home/macierz/s175405/ResearchProject12KASK/birdsDataset')  #Ubuntu 20.04   cluster directory   Quadro RTX 5000 / Quadro RTX 6000

data_ = pd.read_csv('/home/macierz/s175405/ResearchProject12KASK/birdsDataset/birds.csv')
data_.head()

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
        tf.keras.optimizers.RMSprop(learning_rate=0.005),
        #optimizer='Adam',
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
