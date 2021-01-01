import os
import sys
import gc
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from dataloader.datagen import get_generators
from utils.utils import *
from models.effecientnet import efficientNet
from configs.config import CFG

from optimization import optimize

import warnings
warnings.filterwarnings('ignore')

@optimize(5)
def run_model(params, save_model = False):
  train, val, test = get_generators(df_train, df_test)

  STEP_SIZE_TRAIN=train.n//train.batch_size
  STEP_SIZE_VALID=val.n//val.batch_size

  t = time.time()

  tensorboard = TensorBoard(log_dir='logs/{}'.format(
                        f'efficient-net-{t}'))
  
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.25, patience=2, 
                                      verbose=1, mode='auto', min_lr=0.00001)

  early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                                        verbose=1, mode='min')
  
  model = efficientNet(1, params['shape'], params['optimizer'], 
                         params['learning_rate'], params['fine-tune'])
  
  H = model.fit(
        train,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val,
        validation_steps=STEP_SIZE_VALID,
        epochs=params['epochs'],
        callbacks=[early_stop, reduce_lr, tensorboard]
  )
  
  if(save_model):
    print("[INFO] saving model...")
    model.save("efficientNet-{}.model".format(t), save_format="h5")
  
  plot_learning_curve(H, CFG['epochs'])

  
def main():
  pass
  
if __name__=='__main__':
  df = generate_csv('dataloader/data/archive/**/**/*.png')
  df_train, df_test = split_data(df)
  train, val, test = get_generators(df_train, df_test)
  
  loss = run_model()
  
  gc.collect()