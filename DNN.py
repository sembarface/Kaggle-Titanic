import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

from config import Config
config = Config()

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class DNN(nn.Module):
    def __init__(self, input_size, output_size=1, epochs=100, device='cuda', batch_size=32, 
                learning_rate=3e-4):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.preprocessor = make_pipeline(
            SimpleImputer(),
            StandardScaler()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            
            nn.Linear(5, output_size),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        out = self.mlp(x)
        return out
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame, verbose=False):
        
        self.input_size = X_train.shape[1]
        self.to(self.device)
        
        
        self.preprocessor.fit(X_train)
        
        X_train_tensor = torch.tensor(self.preprocessor.transform(X_train), dtype=torch.float32)
        X_val_tensor = torch.tensor(self.preprocessor.transform(X_val), dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1,1)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1,1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        

        
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-8)
        
        
        self.val_losses_ = []
        self.train_losses_ = []
        for epoch in tqdm(range(self.epochs)):
            
            self.train()
            train_loss = 0.0
            
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
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    outputs = self(X_batch)
                    loss = loss_function(outputs, y_batch)
                    
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            self.val_losses_.append(val_loss)
            
            if verbose:
                print((f"Validation Loss: {val_loss:.4f}"))
            
            scheduler.step()
        
        if verbose:
            plt.plot(self.train_losses_,label='train')
            plt.plot(self.val_losses_, label='val')
            plt.legend()
            plt.show()
        
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
        } 
        return checkpoint
    
    def load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(self.device)
        self.input_size = checkpoint['input_size']
        self.preprocessor = checkpoint['preprocessor']
        self.train_losses_ = checkpoint['train_losses']
        self.val_losses_ = checkpoint['val_losses']
            