import random
import pickle
import os

import shap
import numpy as np
import pandas as pd


from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

from config import Config
config = Config()


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import torch


from DNN import DNN
from Solver import Solver
from Dataset import Dataset

from train import train_one_model_CV




def set_seed(seed=config.seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main():
    
    set_seed(config.seed) 
    
    data = pd.read_csv(config.path_to_data, index_col='PassengerId')
    
    if config.use_holdout:
        data, data_test = train_test_split(data, stratify=data['Survived'], test_size=0.2, random_state=config.seed)

    # Test samples for final evaluating
    X_test = data_test.drop('Survived', axis=1)
    y_test = data_test['Survived']
    
    
    dataset = Dataset(data)
    X, y = dataset.get_X(), dataset.get_y()
    
    # Dataset chekpoint saving
    os.makedirs(config.path_to_experiment, exist_ok=True)
    with open(f"{config.path_to_experiment}/dataset_checkpoint.pickle", "wb") as f:
        pickle.dump(dataset, f) 
        
    # Train all models with Cross-Validation
    if config.is_train:
        for model_name in config.models_to_train:
            train_one_model_CV(X, y, model_name)
    
    
    # Pick solver
    solver = Solver()
    pred = solver.predict(X_test)
    print(f'{solver.model_name} test accurasy score = ', accuracy_score(pred, y_test))
    solver.get_plots()
    
    
        

if __name__ == '__main__':
    main()