import os
import json
import subprocess
import time


import numpy as np
import pandas as pd

from config import Config



from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score




from DNN import DNN
from Solver import Solver
from Dataset import Dataset

from train import  run_cv, tune_hyperparameters





def main():
    
    config = Config()
    
    data = pd.read_csv(config.path_to_data, index_col='PassengerId')
    
    if config.use_holdout:
        data, data_test = train_test_split(data, stratify=data['Survived'], test_size=0.2, random_state=config.seed)
        
        # Test samples for final evaluating
        X_test = data_test.drop('Survived', axis=1)
        y_test = data_test['Survived']
    

    
    # Train all models with Cross-Validation
    if config.is_train:
        
        cv = StratifiedKFold(**config.stratified_kfold_parameters)
        dataset = Dataset(data, config.feature_parameters)
        X, y = dataset.get_X(), dataset.get_y()
    
        # Tune hyperparameters
        if config.use_optuna:
            for model_name in config.models_to_train:
                best_params = tune_hyperparameters(X, y, model_name, config)
                config.model_parameters = {
                    **config.model_parameters,
                    model_name: {
                        **config.model_parameters[model_name],
                        **best_params
                    }
                }
            
        for model_name in config.models_to_train:
            
            val_scores, cv_time = run_cv(X, y, model_name, config)
            
            mean_cv_score = np.mean(val_scores)
            print(f'{model_name} CV mean score = {mean_cv_score:.4f}')
            print(f'{model_name} CV scores std = {np.std(val_scores):.4f}')
            print(f'{model_name} CV time = {cv_time:.4f}\n')
            
            # Logging in results.txt
            with open('results.txt', 'a') as f:
                f.write(f'{config.experiment_name} | {model_name} | CV score {mean_cv_score:.4f} | {cv_time:.4f} s\n')
        
        
        # Сertain experiment config
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
             

    # Pick solver
    
    if config.use_holdout:
        solver = Solver(config.path_to_experiment, config.solver_model)
        pred = solver.predict(X_test)
        print(f'{solver.model_name} test accurasy score = ', accuracy_score(pred, y_test))
        
        # Train/val plots
        if config.show_plots:
            solver.get_plots()
            
        
        # Shap beeswarm on train samples
        if config.show_shap:
            X = data.drop('Survived', axis=1)
            solver.get_shap(X)


    
    
    

if __name__ == '__main__':
    main()