CFG = {
  'shape': (32,32,3),
  'batch_size': 256,
  'epochs': 10,
  'learning_rate': 1e-2,
  'training_aug': {
     'rescale': 1/255.0, 'validation_split':0.2
  },
  'test_aug': {
     'rescale':1/255.0
  }
  
}