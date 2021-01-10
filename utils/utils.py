import glob2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from configs.config import CFG
 
plt.style.use("ggplot")


def split_data(df):
    
    X_train, X_test, y_train, y_test = train_test_split(df['id'], df['label'], test_size = 0.2,random_state=42)
    df_train = pd.concat((X_train, y_train), axis = 1)
    df_test = pd.concat((X_test, y_test), axis = 1)
    
    return df_train, df_test


def generate_csv(path, downsample = False):
    
    print('[INFO]::Generating images dataframe...')
    images = glob2.glob(path)
    df = pd.DataFrame({'id': images})
    df['label'] = 0
    df['label'] = df['id'].apply(lambda x: x.split('/')[3])
    if(downsample):
        df_positive = df[df['label'] == '1']
        df_negative = df[df['label'] == '0']
        df_negative = df_negative.sample(n = df_positive.shape[0], random_state = 42)

        df = pd.concat((df_positive, df_negative), axis = 0)
    return df
  

def plot_learning_curve(H, EPOCHS, NAME):
    
    N = np.arange(0, EPOCHS)
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    
    plt.savefig(str(NAME))



def plot_hist(hist):
    
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
