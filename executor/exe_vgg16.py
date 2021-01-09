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

from dataloader.datagen import get_generators_df, get_generators_array
from utils.utils import *
from models.vgg16net import vgg16net
from configs.config import CFG

from optimization import optimize

import warnings
warnings.filterwarnings('ignore')

@optimize(1)
def run_model(params, save_model = False):

    STEP_SIZE_TRAIN=train.n//train.batch_size
    STEP_SIZE_VALID=val.n//val.batch_size

    t = time.time()

    tensorboard = TensorBoard(log_dir='logs/{}'.format(f'vgg16_net_{t}'))
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.25, patience=2,
                                  verbose=1, mode='auto', min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_loss', patience=2,
                               verbose=1, mode='min')

    model = vgg16net(1, params['shape'], params['optimizer'],
                     params['learning_rate'], params['fine-tune'])

    H = model.fit(
        train,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val,
        validation_steps=STEP_SIZE_VALID,
        epochs=params['epochs'],
        callbacks=[early_stop, reduce_lr, tensorboard])
    
    # try:
    #   model.predict_generator(test, callbacks=[tensorboard])
    # except:
    #   print("[Error]: Unable to perform prediction on testset")


    if(save_model):
        print("[INFO] saving model...")        
        model.save("vgg16net_{}.model".format(t), save_format="h5")

    plot_learning_curve(H, CFG['epochs'],f'vgg16_net_{t}.jpg')
    
  
if __name__=='__main__':
    df = generate_csv('dataloader/part1/**/**/*.png', downsample = True)
    df_train, df_test = split_data(df)
    train, val, test = get_generators_df(df_train, df_test)

    loss = run_model()

    gc.collect()