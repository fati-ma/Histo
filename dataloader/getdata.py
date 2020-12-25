from os import listdir
import pandas as pd

def GetData(base_path):
    base_path = base_path
    patient_ids = listdir(base_path)
                         
    columns = ["patient_id",'x','y',"target","path"]
    data_rows = []

    for patient_id in patient_ids:
        for c in [0,1]:
            class_path = base_path + '/' + patient_id + '/' + str(c) + '/'
            imgs = listdir(class_path)
        
        # Extracting Image Paths
            img_paths = [class_path + img  for img in imgs]
        
        # Extracting Image Coordinates
            img_coords = [img.split('_',4)[2:4] for img in imgs]
            x_coords = [int(coords[0][1:]) for coords in img_coords]
            y_coords = [int(coords[1][1:]) for coords in img_coords]

            for (path,x,y) in zip(img_paths,x_coords,y_coords):
                values = [patient_id,x,y,c,path]
                data_rows.append({k:v for (k,v) in zip(columns,values)})
# We create a new dataframe using the list of dicts that we generated above
    data = pd.DataFrame(data_rows)
    data = data.sample(frac=1).reset_index(drop=True)
    
    return data