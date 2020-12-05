#!/usr/bin/env python
# coding: utf-8

# In[10]:


from os import listdir
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder
import random


def sample(base_path,data_size,IMG_DIM):
    base_path = base_path
    patient_ids = listdir(base_path)
    
    class_0_total = 0
    class_1_total = 0
    from pprint import pprint
    for patient_id in patient_ids:
        class_0_files = listdir(base_path + patient_id + '/0')
        class_1_files = listdir(base_path + patient_id + '/1')

        class_0_total += len(class_0_files)
        class_1_total += len(class_1_files) 

    total_images = class_0_total + class_1_total
    
    print(f'Number of patches in Class 0: {class_0_total}')
    print(f'Number of patches in Class 1: {class_1_total}')
    print(f'Total number of patches: {total_images}')
    
    columns = ["patient_id",'x','y',"target","path"]
    data_rows = []
    i = 0
    iss = 0
    isss = 0

# note that we loop through the classes after looping through the 
# patient ids so that we avoid splitting our data into [all class 0 then all class 1]
#'../Adnan/breast cancer/data/IDC_regular_ps50_idx5/10259/0/10259_idx5_x2451_y1501_class0.png',

    for patient_id in patient_ids:
        for c in [0,1]:
            class_path = base_path + patient_id + '/' + str(c) + '/'
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
    print(data.shape)
    data.head()
    
    img_target = [img.split('_')[4].split('.')[0].strip() for img in imgs]
    
    
    all_data = int (data_size / 2)    
    train_data = int (all_data*.6)
    validate_data =int (all_data *.2) 
    test_data =int (all_data *.2) 
    
    positive_data = np.random.choice(data[data.target==1].path.values, size=all_data, replace=False)
    negative_data = np.random.choice(data[data.target==0].path.values, size=all_data, replace=False)

    positive_train = np.random.choice(positive_data, size=train_data, replace=False)
    negative_train = np.random.choice(negative_data, size=train_data, replace=False)

    positive_data = list(set(positive_data) - set(positive_train))
    negative_data = list(set(negative_data) - set(negative_train))


    positive_val = np.random.choice(positive_data, size=validate_data, replace=False)
    negative_val = np.random.choice(negative_data, size=validate_data, replace=False)
    positive_data = list(set(positive_data) - set(positive_val))
    negative_data = list(set(negative_data) - set(negative_val))
  
    positive_test = np.random.choice(positive_data, size=test_data, replace=False)
    negative_test = np.random.choice(negative_data, size=test_data, replace=False)

    print('positive datasets:', positive_train.shape, positive_val.shape, positive_test.shape)
    print('negative datasets:', negative_train.shape, negative_val.shape, negative_test.shape)

    train_files = np.concatenate([positive_train, negative_train])
    validate_files = np.concatenate([positive_val, negative_val])
    test_files = np.concatenate([positive_test, negative_test])
    random.shuffle(train_files)
    random.shuffle(validate_files)
    random.shuffle(test_files)

        
    IMG_DIM = IMG_DIM
    train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
    train_imgs = np.array(train_imgs)

    validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validate_files]
    validation_imgs = np.array(validation_imgs)

    test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
    test_imgs = np.array(test_imgs)


    train_labels = [img.split('_')[7].split('.')[0].strip() for img in train_files]
    validation_labels =[img.split('_')[7].split('.')[0].strip() for img in validate_files]
    test_labels = [img.split('_')[7].split('.')[0].strip() for img in test_files]


    le = LabelEncoder()
    le.fit(train_labels)
    train_labels_enc = le.transform(train_labels)
    validation_labels_enc = le.transform(validation_labels)
    test_labels_enc = le.transform(test_labels)

    return train_imgs,validation_imgs,test_imgs,train_labels_enc,validation_labels_enc,test_labels_enc

        
        
    

