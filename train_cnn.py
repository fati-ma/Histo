import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from models.cnn.CNN import n_layers_cnn
from utils.utils import *

SHAPE = (32,32,3)
BATCH_SIZE = 256
EPOCHS = 20

train_generator = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
test_generator = ImageDataGenerator(rescale=1/255.0)


if __name__ == "__main__"

	df = generate_csv()
	df_train, df_test = split_data(df)

	train = train_generator.flow_from_dataframe(
	    df_train, 
	    x_col = 'id',
	    y_col = 'label',
	    target_size=(SHAPE[0], SHAPE[1]),
	    class_mode = 'binary',
	    shuffle=True,
	    seed = 42,
	    subset="training",
	    classes = ['1', '0'],
	    batch_size = BATCH_SIZE
	)

	val = train_generator.flow_from_dataframe(
	    df_train, 
	    x_col = 'id',
	    y_col = 'label',
	    target_size=(SHAPE[0], SHAPE[1]),
	    class_mode = 'binary',
	    shuffle=True,
	    seed = 42,
	    subset="validation",
	    batch_size = BATCH_SIZE
	)

	test = test_generator.flow_from_dataframe(
	    df_test, 
	    x_col = 'id',
	    y_col = 'label',
	    target_size=(SHAPE[0], SHAPE[1]),
	    class_mode = None,
	    shuffle=False,
	    seed = 42,
	    classes = ['1', '0'],
	    batch_size = BATCH_SIZE
	)

	STEP_SIZE_TRAIN=train.n//train.batch_size
	STEP_SIZE_VALID=val.n//val.batch_size

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
	                                   verbose=1, mode='min', min_lr=0.00001)

	early_stop = EarlyStopping(monitor='val_loss', patience=3, 
	                                        verbose=1, mode='min')
	                              
	model = CNN.n_layers_cnn(SHAPE[0], SHAPE[1], SHAPE[2])
	
	H = model.fit(
	     train,
	     steps_per_epoch=STEP_SIZE_TRAIN,
	     validation_data=val,
	     validation_steps=STEP_SIZE_VALID,
	     epochs=EPOCHS,
	     callbacks=[early_stop, reduce_lr]
	)


	plot_learning_curve(H, EPOCHS):

	print("[INFO] saving model...")
	model.save("histo.model", save_format="h5")