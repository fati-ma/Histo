import pandas as pd
import numpy as np

def DataAug(DataFrame,Augm_size):
    if Augm_size > 78786:
        raise ValueError("maximum Augm_size = 78786")
    
    df = DataFrame[DataFrame.target == 1].reset_index(drop=True)
    DataFrame = DataFrame.drop(DataFrame.index[DataFrame.target==1]).reset_index(drop=True)
    
    df_a = df.sample(n=Augm_size)
    df = df.drop(df_a.index).reset_index(drop=True)
    df_a = df_a.reset_index(drop=True)
    
    
    for i in df_a.index:
        flip = np.random.choice([0,1])
        rotation = np.random.choice([-3,-2,-1,1,2,3])
        imgn = df_a.image[i]
        imgn = np.rot90(imgn,k=rotation)
        imgn = np.flip(imgn,flip)
        patient_id = df_a.patient_id[i]+"_c"
        x = str(df_a.x[i])+"_c"
        y = str(df_a.y[i])+"_c"
        path = "copy of "+ df_a.path[i]
        target = 1
        row = {"patient_id":patient_id,"x":x,"y":y,"path":path,"image":imgn,"target":target}

        df_a = df_a.append(row,ignore_index=True)
        
    Data = pd.concat([DataFrame,df,df_a])
    DataFrame = Data.sample(frac=1).reset_index(drop=True)
    
    return DataFrame