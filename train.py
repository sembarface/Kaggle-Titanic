import pickle
import os
import time
import random


import numpy as np
import pandas as pd

import optuna


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch

from DNN import DNN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index, config, params=None, save_model=True):
  
    print(f"Training {model_name}!")
    
    if params is None:
        params = config.model_parameters[model_name]
    
    cat_columns = config.feature_parameters['cat_columns']
    preprocessor = get_preprocessor(model_name, X_train.columns, cat_columns )
    
    # Logistic Regression
    if model_name == 'logreg':
        model = make_pipeline(
            preprocessor,
            LogisticRegression(**params),
        )
        model.fit(X_train, y_train)
        
    #KNN
    elif model_name == 'KNN':
        model = make_pipeline(
            preprocessor,
            KNeighborsClassifier(**params),
        )
        model.fit(X_train, y_train)
    
    #Decision Tree
    elif model_name == 'decision_tree':
        model = make_pipeline(
            preprocessor,
            DecisionTreeClassifier(**params),
        )
        model.fit(X_train, y_train)

    # Random Forest
    elif model_name == 'random_forest':
        model = make_pipeline(
            preprocessor,
            RandomForestClassifier(**params),
        )
        model.fit(X_train, y_train)
        
    # CatBoost Classifier
    elif model_name == 'catboost':
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_columns)
        
        
    # XGBoost Classifier
    elif model_name == 'xgboost':
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        
    # LightGBM Classifier
    elif model_name == 'lgbm':
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_X=(X_train, X_val), eval_y=(y_train, y_val), 
                    eval_names=['train', 'val'], eval_metric='binary_logloss')
        
        
    # Deep Neural Network   
    elif model_name == 'DNN':
        model = DNN(**params)
        model.fit(X_train, y_train, X_val, y_val, config.feature_parameters)
    
    # Stacking with KNN, Decision Tree and Logistic Regression
    elif model_name == 'stacking_knn_tree_logreg':
        
        estimators = [
            ('knn', make_pipeline(
                get_preprocessor('KNN', X_train.columns, cat_columns),
                KNeighborsClassifier(**config.model_parameters['KNN'])
            )),

            ('tree', make_pipeline(
                get_preprocessor('decision_tree', X_train.columns, cat_columns),
                DecisionTreeClassifier(**config.model_parameters['decision_tree'])
            )),

            ('logreg', make_pipeline(
                get_preprocessor('logreg', X_train.columns, cat_columns),
                LogisticRegression(**config.model_parameters['logreg'])
            ))
        ]

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            **params
        )
        model.fit(X_train, y_train)
    
        
    # Model predictions
    pred = model.predict(X_val)
    score = accuracy_score(pred, y_val)


    # Model saving
    if save_model:
        
        # Path to save models
        save_dir = os.path.join(config.path_to_experiment, f'{model_name}')
        os.makedirs(save_dir, exist_ok=True)
        
        if model_name != 'DNN':
            with open(f"{save_dir}/fold_{fold_index}.pickle", "wb") as f:
                pickle.dump(model, f)
        
        elif model_name == 'DNN':
            checkpoint = model.get_checkpoint()
            with open(f"{save_dir}/fold_{fold_index}.pickle", "wb") as f:
                pickle.dump(checkpoint, f)
    
    return score


def get_preprocessor(model_name, columns, cat_columns):
    
    if model_name in ['catboost', 'xgboost', 'lgbm', 'DNN']:
        return None

    numerical = [
        col for col in columns
        if col not in cat_columns
    ]

    # Fill nan
    numeric_steps = [SimpleImputer(strategy='median')]

    # Scale-sensitive models 
    if model_name in ['logreg', 'KNN']:
        numeric_steps.append(StandardScaler())

    return ColumnTransformer([
        ('num', make_pipeline(*numeric_steps), numerical),
        ('cat', make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore')
        ), cat_columns)
    ])

def run_cv(X, y, model_name, config, params=None, save_models=True):
    
    # Cross-Validation strategy
    cv = StratifiedKFold(**config.stratified_kfold_parameters)
    
    val_scores = []
     
    start_time = time.perf_counter()
    for fold_index, (train_indices, val_indices) in enumerate(cv.split(X, y)):
        print(f'Validation {fold_index+1} fold started!')
        
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
        
        
        set_seed(config.seed)
        
        score = train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index, config, params, save_models)

        val_scores.append(score)
    
    end_time = time.perf_counter()
    cv_time = end_time - start_time
    
    return val_scores, cv_time
      
# Optuna    
def suggest_parameters(trial, search_space):
    params = {}

    methods = {
        'int': trial.suggest_int,
        'float': trial.suggest_float,
        'categorical': trial.suggest_categorical
    }

    for model_name, settings in search_space.items():
        param_type = settings['type']
        kwargs = {k: v for k, v in settings.items() if k != 'type'}

        params[model_name] = methods[param_type](model_name, **kwargs)

    return params

# Optuna
def tune_hyperparameters(X, y, model_name, config):

    def objective(trial):
        suggested_params = suggest_parameters(
            trial,
            config.search_spaces[model_name]
        )

        params = {
            **config.model_parameters[model_name],
            **suggested_params
        }

        scores, _ = run_cv(
            X, y, model_name, config,
            params=params,
            save_models=False
        )

        return float(np.mean(scores))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=config.seed
        )
    )

    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        show_progress_bar=True,
        n_jobs=-1,
    )

    print(f'{model_name} best CV accuracy: {study.best_value:.4f}')
    print(f'{model_name} best parameters: {study.best_params}')

    return study.best_params