import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class CNN:
  @staticmethod
  def n_layers_cnn(width, height, channels, filters = (16,32,32,32,64,64)):
    inputShape = (height, width, channels)
    input = Input(shape=inputShape)
    x = input
    for idx, filter in enumerate(filters):
      if ((idx+1)%3 == 0):
        x = Dropout(0.2)(x)

      x = Conv2D(filter, (3, 3), strides=2, padding="same")(x)
      x = LeakyReLU(alpha=0.2)(x)
      x = BatchNormalization(axis=-1)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Dense(256)(x)
    out = Dense(2)(x)
    
    model = Model(input, out, name=f"{len(filters)}_layers_cnn")

    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])
    
    return model