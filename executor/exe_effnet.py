import os
import sys
import gc

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from dataloader.datagen import get_generators
from utils.utils import *
from models.effecientnet import efficientNet
from configs.config import CFG

def main():
  df = generate_csv('dataloader/data/archive/**/**/*.png')
  df_train, df_test = split_data(df)
  train, val, test = get_generators(df_train, df_test)

  STEP_SIZE_TRAIN=train.n//train.batch_size
  STEP_SIZE_VALID=val.n//val.batch_size

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                      verbose=1, mode='min', min_lr=0.00001)

  early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                                          verbose=1, mode='min')
                                
  model = efficientNet(2, CFG['shape'])

  H = model.fit(
        train,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val,
        validation_steps=STEP_SIZE_VALID,
        epochs=CFG['epochs'],
        callbacks=[early_stop, reduce_lr]
  )

  print("[INFO] saving model...")
  model.save("histo.model", save_format="h5")
  
  plot_learning_curve(H, CFG['epochs'])

  
if __name__=='__main__':
  main()
  gc.collect()