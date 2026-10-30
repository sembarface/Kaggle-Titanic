import copy

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR



class DNN(nn.Module):
    def __init__(self, output_size=1, epochs=100, device='cuda', batch_size=32, 
                learning_rate=3e-4, optimizer_name='Adam'):
        super().__init__()
        
        self.output_size = output_size
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        
        
        
    def _build_model(self, input_size):
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            
            nn.Linear(5, self.output_size),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        out = self.mlp(x)
        return out
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame, feature_parameters, verbose=False):
        
        categorical = feature_parameters['cat_columns']

        numerical = [
            col for col in X_train.columns
            if col not in categorical
        ]


        self.preprocessor = ColumnTransformer([
            ('num', make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler()
            ), numerical),

            ('cat', make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                )
            ), categorical)
        ])
        
        
        self.preprocessor.fit(X_train)
        
        # Prepare data
        X_train_tensor = torch.tensor(self.preprocessor.transform(X_train), dtype=torch.float32)
        X_val_tensor = torch.tensor(self.preprocessor.transform(X_val), dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1,1)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1,1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        
        self.input_size = X_train_tensor.shape[1]
        self._build_model(self.input_size)
        self.to(self.device)
        

        
        loss_function = torch.nn.BCELoss()
        optimizer = get_optimizer(self.optimizer_name, params=self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        
        self.val_losses_ = []
        self.train_losses_ = []
        self.best_val_loss_ = torch.inf
        best_state = None
        
        # Training
        for epoch in tqdm(range(self.epochs)):
            
            self.train()
            train_loss = 0.0
            # Train loop
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self(X_batch)
                loss = loss_function(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            self.train_losses_.append(train_loss)
            
            if verbose:
                print(f"Training Loss: {train_loss:.4f}")
            
            
            self.eval()
            val_loss = 0.0
            # Validation loop
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    outputs = self(X_batch)
                    loss = loss_function(outputs, y_batch)
                    
                    val_loss += loss.item()
                
            val_loss /= len(val_loader)
            self.val_losses_.append(val_loss)
            
            # Remember best_state
            if val_loss < self.best_val_loss_:
                self.best_epoch_ = epoch + 1
                self.best_val_loss_ = val_loss
                best_state = copy.deepcopy(self.state_dict())
            
            if verbose:
                print((f"Validation Loss: {val_loss:.4f}"))
            
            scheduler.step()
        
        if verbose:
            print(f'Best epoch: {self.best_epoch}')
            plt.plot(self.train_losses_,label='train')
            plt.plot(self.val_losses_, label='val')
            plt.legend()
            plt.show()

        self.load_state_dict(best_state)
        return self
    
    def predict_proba(self, X: pd.DataFrame):
        self.eval()
        
        with torch.no_grad():
            
            X_tensor = torch.tensor(self.preprocessor.transform(X), dtype=torch.float32).to(self.device)
            probabilities = self(X_tensor).cpu().numpy()
            
            return np.hstack([1- probabilities, probabilities])
    
    def predict(self, X: pd.DataFrame):
        
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def get_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'preprocessor': self.preprocessor,
            'train_losses': self.train_losses_,
            'val_losses' : self.val_losses_,
            'best_epoch': self.best_epoch_
        } 
        return checkpoint
    
    def load_checkpoint(self, checkpoint):
        self.input_size = checkpoint['input_size']
        self._build_model(self.input_size)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(self.device)
        
        self.preprocessor = checkpoint['preprocessor']
        self.train_losses_ = checkpoint['train_losses']
        self.val_losses_ = checkpoint['val_losses']
        self.best_epoch_ = checkpoint['best_epoch']


def get_optimizer(optimizer_name, params, lr):
    if optimizer_name == 'Adam':
        optimizer = Adam(params, lr=lr)
    
    elif optimizer_name == 'SGD':
        optimizer = SGD(params, lr=lr)
    
    return optimizer