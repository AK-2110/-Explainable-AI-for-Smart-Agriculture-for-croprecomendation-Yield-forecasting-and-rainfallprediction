import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def fit_transform_tabular(self, df, numeric_cols, categorical_cols=None, target_col=None):
        """
        Preprocesses tabular data for Crop Recommendation.
        """
        df_processed = df.copy()
        
        # Numeric Scaling
        if numeric_cols:
            scaler = StandardScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['tabular_num'] = scaler
            
        # Categorical Encoding
        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df[col])
                self.encoders[col] = le
                
        # Target Encoding
        if target_col and target_col in df.columns:
            le_target = LabelEncoder()
            df_processed[target_col] = le_target.fit_transform(df[target_col])
            self.encoders[target_col] = le_target
            
        return df_processed
        
    def create_sequences(self, data, seq_length, target_col_idx):
        """
        Creates sequences for Time-Series forecasting (Yield/Rainfall).
        """
        X, y = [], []
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        for i in range(len(data_array) - seq_length):
            X.append(data_array[i:(i + seq_length)])
            y.append(data_array[i + seq_length, target_col_idx])
            
        return np.array(X), np.array(y)
        
    def preprocess_timeseries(self, df, numeric_cols, target_col):
        """
        Scales time-series data.
        """
        scaler = MinMaxScaler()
        # Scale all columns including target for simplicity in sequence generation, 
        # or separate if needed. Here we scale everything.
        data_scaled = scaler.fit_transform(df[numeric_cols + [target_col]])
        self.scalers[target_col] = scaler
        
        return data_scaled, scaler
