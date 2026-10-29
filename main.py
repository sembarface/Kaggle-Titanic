import random
import os
import json
import subprocess
import time

import shap
import numpy as np
import pandas as pd

from config import Config



from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score



import torch


from DNN import DNN
from Solver import Solver
from Dataset import Dataset

from train import  train_one_model




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main():
    
    config = Config()
    
    data = pd.read_csv(config.path_to_data, index_col='PassengerId')
    
    if config.use_holdout:
        data, data_test = train_test_split(data, stratify=data['Survived'], test_size=0.2, random_state=config.seed)
        
        # Test samples for final evaluating
        X_test = data_test.drop('Survived', axis=1)
        y_test = data_test['Survived']
    
    # Saving certain experiment config
    if config.is_train:
        
        experiment_config = config.to_dict()
        
        # Trying to save commit hash
        try:
            experiment_config['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                text=True
            ).strip()
        except(subprocess.CalledProcessError, FileNotFoundError):
            print('No Git')
            experiment_config['git_commit'] = None
            
            
        os.makedirs(config.path_to_experiment, exist_ok=True)
        # Saving experiment config
        with open(f'{config.path_to_experiment}/config.json', 'w') as f:
            json.dump(experiment_config, f, indent=4)
    
    

    # Train all models with Cross-Validation
    if config.is_train:
        # Cross-Validation strategy
        cv = StratifiedKFold(**config.stratified_kfold_parameters)
        
    
        for model_name in config.models_to_train:
            
            val_scores = []
            start_time = time.perf_counter()
            
            for fold_index, (train_indices, val_indices) in enumerate(cv.split(data, data['Survived'])):
                print(f'Validation {fold_index+1} fold started!')
                
                dataset = Dataset(data, config.feature_parameters)
                X, y = dataset.get_X(), dataset.get_y()
                
                X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
                X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
                
                
                set_seed(config.seed)
                
                score = train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index, config)

                val_scores.append(score)
            
            end_time = time.perf_counter()
            cv_score = sum(val_scores)/len(val_scores)
            cv_time = end_time - start_time
            print(f'{model_name} CV mean score = {cv_score:.4f}')
            print(f'{model_name} CV scores std = {np.std(val_scores):.4f}')
            print(f'{model_name} CV time = {cv_time:.4f}\n')
            
            # Logging in results.txt
            with open('results.txt', 'a') as f:
                f.write(f'{config.experiment_name} | {model_name} | CV score {cv_score:.4f} | {cv_time:.4f} s\n')
                                    
             

    # Pick solver
    
    if config.use_holdout:
        
        solver = Solver()
        pred = solver.predict(X_test)
        print(f'{solver.model_name} test accurasy score = ', accuracy_score(pred, y_test))
        # solver.get_plots()
    
    
    
    

if __name__ == '__main__':
    main()