class Config:
    """Store experiment settings, model parameters, and search spaces."""
    
    seed = 228

    experiment_name = '---'
    
    models_to_train = [
        'logreg',
        'KNN',
        'decision_tree',
        'random_forest',
        'catboost',
        'xgboost',
        'lgbm',
        'DNN',
        'stacking_knn_tree_logreg'
    ]
    
    solver_model = 'catboost'
    get_submission = True
    
    use_holdout = False

    is_train = True
    
    use_optuna = False
    optuna_n_trials = 30
    
    show_plots = False
    show_shap = False
    
    path_to_experiment = f"experiments/{experiment_name}"
    path_to_data = f"data/train.csv"
    path_to_submissions = f'submissions'
    
    
    feature_parameters = {
        'cat_columns': ['Sex', 'Embarked', 'Title'],
        'drop_columns': ['Name', 'Ticket', 'Cabin'],
        'add_title': True,
        'add_family_size': True,
        'add_is_alone': True,
    }
    
    stratified_kfold_parameters = {
        'n_splits': 5,
        'shuffle': True,
        'random_state': seed
    }
    
    model_parameters = {
        'logreg': {
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': seed,
        },

        'KNN': {
            'n_neighbors': 7,
            'weights': 'distance',
            'metric': 'minkowski',
            'p': 2,
        },

        'decision_tree': {
            'criterion': 'gini',
            'max_depth': 4,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': seed,
        },

        'random_forest': {
            'n_estimators': 300,
            'criterion': 'gini',
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': seed,
        },

        'catboost': {
            'iterations': 500,
            'depth': 4,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': seed,
            'one_hot_max_size': 10,
            'early_stopping_rounds': 50,
            'use_best_model': True,
            'verbose': 0,
        },

        'xgboost': {
            'n_estimators': 500,
            'max_depth': 3,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_lambda': 2.0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': seed,
            'enable_categorical':True,
            'early_stopping_rounds': 50,
        },

        'lgbm': {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': -1,
            'min_child_samples': 15,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.9,
            'reg_lambda': 1.0,
            'verbosity': -1,
            'random_state': seed,
            'early_stopping_rounds': 50,
        },

        'DNN': {
            'device': 'cpu',
            'output_size': 1,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 3e-2,
            'optimizer_name': 'Adam',
        },
        
        'stacking_knn_tree_logreg':{
            'cv': 5,
            'stack_method': 'predict_proba'
        }
    }
    
    search_spaces = {
        'logreg': {
            'C': {
                'type': 'float',
                'low': 1e-3,
                'high': 100,
                'log': True
            },
        },

        'KNN': {
            'n_neighbors': {
                'type': 'int',
                'low': 3,
                'high': 25
            },
            'weights': {
                'type': 'categorical',
                'choices': ['uniform', 'distance']
            },
            'p': {
                'type': 'categorical',
                'choices': [1, 2]
            },
        },

        'decision_tree': {
            'max_depth': {
                'type': 'int',
                'low': 2,
                'high': 12
            },
            'min_samples_leaf': {
                'type': 'int',
                'low': 1,
                'high': 15
            },
        },

        'random_forest': {
            'max_depth': {
                'type': 'int',
                'low': 3,
                'high': 12
            },
            'min_samples_leaf': {
                'type': 'int',
                'low': 1,
                'high': 10
            },
            'max_features': {
                'type': 'categorical',
                'choices': ['sqrt', 0.5, None]
            },
        },

        'catboost': {
            'iterations': {
                'type': 'int',
                'low': 300,
                'high': 600,
                'step': 100
            },
            'depth': {
                'type': 'int',
                'low': 3,
                'high': 7
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.001,
                'high': 0.1,
                'log': True
            },
            'l2_leaf_reg': {
                'type': 'float',
                'low': 1.0,
                'high': 10.0,
                'log': True
            },
            'random_strength': {
                'type': 'float',
                'low': 0.0,
                'high': 2.0
            },
        },

        'xgboost': {
            'n_estimators': {
                'type': 'int',
                'low': 300,
                'high': 600,
                'step': 100
            },
            'max_depth': {
                'type': 'int',
                'low': 2,
                'high': 6
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.001,
                'high': 0.1,
                'log': True
            },
            'min_child_weight': {
                'type': 'int',
                'low': 1,
                'high': 10
            },
            'reg_lambda': {
                'type': 'float',
                'low': 0.1,
                'high': 10.0,
                'log': True
            },
        },

        'lgbm': {
            'n_estimators': {
                'type': 'int',
                'low': 300,
                'high': 600,
                'step': 100
            },
            'num_leaves': {
                'type': 'int',
                'low': 5,
                'high': 31
            },
            'min_child_samples': {
                'type': 'int',
                'low': 5,
                'high': 40
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.001,
                'high': 0.1,
                'log': True
            },
            'reg_lambda': {
                'type': 'float',
                'low': 0.0,
                'high': 5.0
            },
        },

        'DNN': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-1,
                'log': True
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64]
            },
            'epochs': {
                'type': 'int',
                'low': 100,
                'high': 300,
                'step': 100
            },
            "optimizer_name": {
                'type': 'categorical',
                'choices': ["Adam", "SGD"]
            }
        },

        'stacking_knn_tree_logreg': {
            'cv': {
                'type': 'categorical',
                'choices': [3, 5]
            },
        },
    }
    
    def to_dict(self):
        """Convert the experiment configuration into a dictionary."""
        
        return {
            'experiment_name': self.experiment_name,
            'use_optuna': self.use_optuna,
            'optuna_n_trials': self.optuna_n_trials if self.use_optuna else None,
            'seed': self.seed,
            'feature_parameters': self.feature_parameters,
            'model_parameters': {
                name: self.model_parameters[name]
                for name in self.models_to_train
            },
            'stratified_kfold_parameters': self.stratified_kfold_parameters,
            'use_holdout': self.use_holdout,
            'models_to_train': self.models_to_train,
            'solver_model': self.solver_model
        }