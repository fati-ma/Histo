from dataloader.getdata import GetData
from dataloader.samplebaldata import SampleBalData
from dataloader.dataaugm import DataAug
from dataloader.splitdata import SaplitData
from configs.CFG import *

df = GetData(base_path)
df_ag = DataAug(df,Augm_size)
sdf = SampleBalData(df_ag,data_size)
X_train, y_train, X_val, y_val, X_test, y_test = SaplitData(sdf,train_size,val_size,test_size)
