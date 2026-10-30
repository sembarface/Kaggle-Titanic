import pickle
import os
import json

import shap
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

from catboost import Pool
from xgboost import DMatrix

from Dataset import Dataset
from DNN import DNN



# Inference
class Solver:
    def __init__(self, path_to_experiment, model_name):
        
        self.model_name = model_name
        
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
                    model = DNN()
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
    
    def get_plots(self, fold =0):

        if self.model_name == 'catboost':
            train_result = self.models[fold].get_evals_result()['learn']['Logloss']
            val_result = self.models[fold].get_evals_result()['validation']['Logloss']

        elif self.model_name == 'xgboost':
            train_result = self.models[fold].evals_result()['validation_0']['logloss']
            val_result = self.models[fold].evals_result()['validation_1']['logloss']

        elif self.model_name == 'lgbm':
            train_result = self.models[fold].evals_result_['train']['binary_logloss']
            val_result = self.models[fold].evals_result_['val']['binary_logloss']

        elif self.model_name == 'DNN':
            train_result = self.models[fold].train_losses_
            val_result = self.models[fold].val_losses_

        else:
            print(f'No plots with {self.model_name}')
            return


        plt.plot(train_result, label=f'train fold {fold}')
        plt.plot(val_result, label=f'val fold {fold}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



    def get_shap(self, X, fold = 0):
        
        model = self.models[fold]
        X = self.dataset.preprocessing(X)
        
        if self.model_name == 'catboost':
            pool = Pool(
                X,
                cat_features=model.get_cat_feature_indices()
            )

            values = model.get_feature_importance(
                data=pool,
                type='ShapValues'
            )

        elif self.model_name == 'xgboost':
            dmatrix = DMatrix(X, enable_categorical=True)

            booster = model.get_booster()

            best_iteration = getattr(model, 'best_iteration', None)

            values = booster.predict(
                dmatrix,
                pred_contribs=True,
                iteration_range=(
                    0, best_iteration + 1
                ) if best_iteration is not None else (0, 0)
            )

        elif self.model_name == 'lgbm':
            values = model.predict(
                X,
                pred_contrib=True
            )

        else:
            print(f'SHAP is not supported for {self.model_name}')
            return


        explanation = shap.Explanation(
            values=values[:, :-1],
            base_values=values[:, -1],
            data=X.to_numpy(),
            feature_names=X.columns.tolist()
        )
        
        shap.plots.beeswarm(explanation)
        
        return explanation