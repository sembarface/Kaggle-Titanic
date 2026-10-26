import random
import pickle
import os
import time


import numpy as np
import pandas as pd

from config import Config
config = Config()


from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier




from DNN import DNN

def train_one_model_CV(X, y, model_name):
    # Cross-Validation strategy
    cv = StratifiedKFold(**config.stratified_kfold_parameters)
    
        
    # Path to save models
    save_dir = os.path.join(config.path_to_experiment, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    val_scores = []
    start_time = time.perf_counter()
    
    for fold_index, (train_indices, val_indices) in enumerate(cv.split(X, y)):
        print(f'Validation {fold_index+1} fold started!')
        
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
        
        params = config.model_parameters[model_name]
    
        # Logistic Regression
        if model_name == 'logreg':
            model = make_pipeline(
                SimpleImputer(),
                StandardScaler(),
                LogisticRegression(**params),
            )
            model.fit(X_train, y_train)
            
        #KNN
        elif model_name == 'KNN':
            model = make_pipeline(
                SimpleImputer(),
                StandardScaler(),
                KNeighborsClassifier(**params),
            )
            model.fit(X_train, y_train)
        
        #Decision Tree
        elif model_name == 'decision_tree':
            model = make_pipeline(
                SimpleImputer(),
                DecisionTreeClassifier(**params),
            )
            model.fit(X_train, y_train)

        # Random Forest
        elif model_name == 'random_forest':
            model = make_pipeline(
                SimpleImputer(),
                RandomForestClassifier(**params),
            )
            model.fit(X_train, y_train)
            
        # CatBoost Classifier
        elif model_name == 'catboost':
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            
        # XGBoost Classifier
        elif model_name == 'xgboost':
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
            
        # LightGBM Classifier
        elif model_name == 'lgbm':
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_X=(X_train, X_val), eval_y=(y_train, y_val), 
                      eval_names=['train', 'val'], eval_metric='binary_logloss')
            
            
        # Deep Neural Network   
        elif model_name == 'DNN':
            model = DNN(X_train.shape[1], **params)
            model.fit(X_train, y_train, X_val, y_val)
            
        
        # Model predictions
        pred = model.predict(X_val)
                    
        fold_score = accuracy_score(pred, y_val)
        val_scores.append(fold_score)


        # Model saving
        if model_name in ["logreg", 'KNN', 'decision_tree', 'random_forest', 'catboost', 'xgboost', 'lgbm']:
            with open(f"{save_dir}/fold_{fold_index}.pickle", "wb") as f:
                pickle.dump(model, f)
        
        elif model_name == 'DNN':
            checkpoint = model.get_checkpoint()
            with open(f"{save_dir}/fold_{fold_index}.pickle", "wb") as f:
                pickle.dump(checkpoint, f)
            
            
    
    end_time = time.perf_counter()         
    cv_score = sum(val_scores)/len(val_scores)
    cv_time = end_time - start_time
    print(f'{model_name} CV mean score = {cv_score:.4f}')
    print(f'{model_name} CV scores std = {np.std(val_scores):.4f}')
    print(f'{model_name} CV time = {cv_time:.4f}')
    
    # Logging in results.txt
    with open('results.txt', 'a') as f:
        f.write(f'{config.experiment_name} | {model_name} | CV score {cv_score:.4f} | {cv_time:.4f} s\n')
                        
    
         
    