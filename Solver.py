import pickle
import os
import json

import shap
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

from config import Config
config = Config()

from Dataset import Dataset
from DNN import DNN



# Inference
class Solver:
    def __init__(self, model_name=config.solver_model):
        
        self.model_name = model_name
        
        path_to_experiment = config.path_to_experiment
        path_to_models = os.path.join(path_to_experiment, model_name)
        
        # Experiment config loading
        with open(f'{path_to_experiment}/config.json') as f:
            self.experiment_config = json.load(f)
        
        self.dataset = Dataset(feature_parameters=self.experiment_config['feature_parameters'])
        
        # Models loading
        self.models = []
        for fold in sorted(os.listdir(path_to_models)):
            if not fold.startswith('fold_') or not fold.endswith('.pickle'):
                continue
            
            with open(f'{path_to_models}/{fold}', 'rb') as f:
                if model_name != 'DNN':
                    model = pickle.load(f)
                    

                elif model_name == 'DNN':
                    checkpoint = pickle.load(f)
                    model = DNN(input_size=checkpoint['input_size'])
                    model.load_checkpoint(checkpoint)
                    model.eval()
                    
                self.models.append(model)
    
    def predict_proba(self, X):
        X = self.dataset.preprocessing(X)
        probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            probabilities.append(proba)
        mean_probabilities = np.mean(probabilities, axis=0)
        
        return mean_probabilities

    def predict(self, X):
        mean_probabilities = self.predict_proba(X)
        mean_predictions = np.argmax(mean_probabilities, axis=1)
        
        return mean_predictions
    
    def get_plots(self):
        if self.model_name in ["logreg", 'KNN', 'decision_tree', 'random_forest']:
            print('No plots with sklearn models')
            return
        
        if self.model_name == 'catboost':
            mean_train_results = np.mean([model.get_evals_result()['learn']['Logloss'] for model in self.models], axis=0)
            mean_val_results = np.mean([model.get_evals_result()['validation']['Logloss'] for model in self.models], axis=0)
        
        elif self.model_name == 'xgboost':
            mean_train_results = np.mean([model.evals_result()['validation_0']['logloss'] for model in self.models], axis=0)
            mean_val_results = np.mean([model.evals_result()['validation_1']['logloss'] for model in self.models], axis=0)
        
        elif self.model_name == 'lgbm':
            mean_train_results = np.mean([model.evals_result_['train']['binary_logloss'] for model in self.models], axis=0)
            mean_val_results = np.mean([model.evals_result_['val']['binary_logloss'] for model in self.models], axis=0)
        
        elif self.model_name == 'DNN':
            mean_train_results = np.mean([model.train_losses_ for model in self.models], axis=0)
            mean_val_results = np.mean([model.val_losses_ for model in self.models], axis=0)
        
        plt.plot(mean_train_results, label='train')
        plt.plot(mean_val_results, label='val')
        plt.legend()
        plt.show()
            
            
        
            
        
