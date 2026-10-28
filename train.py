import pickle
import os


import numpy as np
import pandas as pd

from config import Config
config = Config()



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from DNN import DNN


def train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index):
  
    # Path to save models
    save_dir = os.path.join(config.path_to_experiment, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training {model_name}!")
    
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
        model.fit(X_train, y_train, X_val, y_val)
        
    # Model predictions
    pred = model.predict(X_val)
    score = accuracy_score(pred, y_val)


    # Model saving
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
    