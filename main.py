from dataloader.getdata import GetData
from dataloader.samplebaldata import SampleBalData
from dataloader.splitdata import SaplitData
from configs.CFG import *

df = GetData(base_path)
sdf = SampleBalData(df,data_size)
dftr, dfval, dftst = SaplitData(sdf,train_size,val_size,test_size)