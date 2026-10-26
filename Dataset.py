import pandas as pd


class Dataset:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data

        self.features = self.data.drop('Survived', axis =1)
        self.target = self.data['Survived']
            
    
    def preprocessing(self, features: pd.DataFrame) -> pd.DataFrame:
        
        df = features.copy()
        
        df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Fare'])
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        df['Sex'] = (df['Sex'] == 'male')
        
        return df

    def get_X(self) -> pd.DataFrame:
        
        X = self.preprocessing(self.features)
        return X
    
    def get_y(self) -> pd.Series:
        
        y = self.target.copy()
        return y
    