CFG = {
  'shape': (96,96,3),
  'batch_size': 32,
  'epochs': 5,
  
  #available optimizers are 'Adam' and 'RMSprop'
  'optimizer': ['Adam'],
  'learning_rate': (1e-3, 3e-3),
  'training_aug': {
     'rescale': 1/255.0, 
     'rotation_range': 90,
     'horizontal_flip': True,
     'vertical_flip': True,
     'validation_split':0.2,
  },
  'test_aug': {
     'rescale':1/255.0
  },
  
  'fine-tune': [False]
}