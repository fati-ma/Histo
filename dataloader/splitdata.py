import pandas as pd
import numpy as np

def SaplitData(data_frame,train_size,val_size,test_size):
    
    val_s = val_size / (val_size + test_size)
    val_s = round(val_s,2)
    
    pos_df = data_frame[data_frame.target == 1]
    neg_df = data_frame[data_frame.target == 0]
    
    train_pos = pos_df.sample(frac=train_size)
    pos_df = pos_df.drop(train_pos.index)
    
    val_pos = pos_df.sample(frac=val_s)
    test_pos = pos_df.drop(val_pos.index)
    
    train_neg = neg_df.sample(frac=train_size)
    neg_df = neg_df.drop(train_neg.index)
    
    val_neg = neg_df.sample(frac=val_s)
    test_neg = neg_df.drop(val_neg.index)
    
    train_df = pd.concat([train_pos,train_neg])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    X_train=np.array(train_df.image)
    Y_train=np.array(train_df.target)
    
    val_df = pd.concat([val_pos,val_neg])
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    X_val=np.array(val_df.image)
    Y_val=np.array(val_df.target)
    
    test_df = pd.concat([test_pos,test_neg])
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    X_test=np.array(test_df.image)
    Y_test=np.array(test_df.target)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
