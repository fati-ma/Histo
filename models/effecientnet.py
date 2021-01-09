import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers.experimental import preprocessing
from configs.config import CFG

import gc

def efficientNet(num_classes, img_size, optimizer, lr, fine_tune=False):
    model = keras.models.Sequential([
        EfficientNetB5(include_top=False, input_shape=img_size, weights="imagenet"),
        layers.GlobalAveragePooling2D(name="avg_pool"),
        layers.Dense(num_classes, activation="sigmoid", name="pred")
    
    ])

    # if(fine_tune):
    #     # unfreeze the pretrained weights
    #     for layer in model.layers[-20:]:
    #         if not isinstance(layer, layers.BatchNormalization):
    #             layer.trainable = True

    # Rebuild top
    # x = 
    # outputs = 

    # # Compile
    # model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    
    if(optimizer == 'Adam'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif(optimizer == 'RMSprop'):
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    
    print(model.summary())
    
    return model
    
  
