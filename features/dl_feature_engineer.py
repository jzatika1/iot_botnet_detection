from .feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DLFeatureEngineer(FeatureEngineer):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.label_encoding = self.config['feature_engineering']['dl']['label_encoding']

    def engineer_features(self, X, y, dataset_name):
        self.logger.info("Starting DL feature engineering process")
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        X = self.remove_problematic_rows(X)
        y = y.loc[X.index]
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"DL Feature engineering completed. Output shape: {X.shape}")
        return X.values, y_encoded

    def encode_categorical_features(self, X):
        for column in X.columns:
            if X[column].dtype == 'object':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        return X

    def inverse_transform_features(self, X_array):
        X = pd.DataFrame(X_array, columns=self.label_encoders.keys())
        for column, encoder in self.label_encoders.items():
            X[column] = encoder.inverse_transform(X[column].astype(int))
        return X