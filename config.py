class Config:
    seed = 228

    experiment_name = 'default_tuned'
    models_to_train = ["logreg", 'KNN', 'decision_tree', 'random_forest', 
                       'catboost', 'xgboost', 'lgbm', 'DNN']
    
    
    
    
    path_to_experiment = f"experiments/{experiment_name}"
    
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
        },

        'DNN': {
            'output_size': 1,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 3e-4,
        },
    }