import pandas as pd

def SampleBalData(data_frame,data_size):
    if data_size > 78786:
        raise ValueError("maximum possible balanced data_size = 78786")
        
    half_data = int (data_size / 2)   
    
    pos_df = data_frame[data_frame.target == 1]
    sub_pos_df = pos_df.sample(n=half_data)
    
    neg_df = data_frame[data_frame.target == 0]  
    sub_neg_df = neg_df.sample(n=half_data)

    sub_df = pd.concat([sub_pos_df,sub_neg_df])
    sub_bal_df = sub_df.sample(frac=1).reset_index(drop=True)
    

    return sub_bal_df
