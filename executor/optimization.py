import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import optuna
from configs.config import CFG

class optimize:
  """
  Decorator class for automating hyperparamters tuning using OpTuna
  """
  
  def __init__(self, n_trials = 5, direction='minimize'):  
    self.n_trials = n_trials
    self.direction = direction
    self.executor = None
        
  def __call__(self, executor):
    self.executor = executor
    
    def wrapper(*args, **kwargs):
      self.study = optuna.create_study(direction=self.direction)
      return self.study.optimize(self._objective, n_trials=self.n_trials)
    
    return wrapper
        
  def _objective(self, trail):
    params = {
      'shape': CFG['shape'],
      'batch_size': CFG['batch_size'],
      'epochs': CFG['epochs'],
      
      'optimizer': trail.suggest_categorical('optimizer', CFG['optimizer']),
      
      'learning_rate': trail.suggest_loguniform(
          'learning_rate', CFG['learning_rate'][0], CFG['learning_rate'][1]),
    
      'training_aug': CFG['training_aug'],
      'test_aug': CFG['test_aug'],
      
      'fine-tune': trail.suggest_categorical('fine-tune', CFG['fine-tune'])
    }
    
    loss = self.executor(params)
    return loss