import pandas as pd


class Dataset:
    
    def __init__(self, data: pd.DataFrame = None, feature_parameters: dict =None):
        
        self.feature_parameters = feature_parameters
        
        if data is not None:
            self.data = data
            self.features = self.data.drop('Survived', axis =1)
            self.target = self.data['Survived']
            

    def preprocessing(self, features: pd.DataFrame) -> pd.DataFrame:
        
        df = features.copy()
        
        df = df.drop(columns=self.feature_parameters['drop_columns'])
        
        # Categorical features
        for col in self.feature_parameters['cat_columns']:
            df[col] = df[col].fillna('Unknown').astype('category')
        
        return df

    def get_X(self) -> pd.DataFrame:
        
        X = self.preprocessing(self.features)
        return X
    
    def get_y(self) -> pd.Series:
        
        y = self.target.copy()
        return y
    