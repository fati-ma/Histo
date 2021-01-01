import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils.utils import *
from configs.config import CFG

SHAPE = CFG['shape']
BATCH_SIZE = CFG['batch_size']
EPOCHS = CFG['epochs']

def get_generators(df_train, df_test):
  train_generator = ImageDataGenerator(**CFG['training_aug'])
  test_generator = ImageDataGenerator(**CFG['test_aug'])

  train = train_generator.flow_from_dataframe(
      df_train, 
      x_col = 'id',
      y_col = 'label',
      target_size=(SHAPE[0], SHAPE[1]),
      class_mode = 'binary',
      shuffle=True,
      seed = 42,
      subset="training",
      batch_size = BATCH_SIZE
  )

  val = train_generator.flow_from_dataframe(
      df_train, 
      x_col = 'id',
      y_col = 'label',
      target_size=(SHAPE[0], SHAPE[1]),
      class_mode = 'binary',
      shuffle=False,
      seed = 42,
      subset="validation",
      batch_size = BATCH_SIZE
  )

  test = test_generator.flow_from_dataframe(
      df_test, 
      x_col = 'id',
      y_col = 'label',
      target_size=(SHAPE[0], SHAPE[1]),
      class_mode = 'binary',
      shuffle=False,
      seed = 42,
      batch_size = BATCH_SIZE
  )

  return train, val, test
