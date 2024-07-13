import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logger import Logger
from abc import ABC, abstractmethod

class FeatureEngineer(ABC):
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or Logger.setup('FeatureEngineer')
        self.label_encoder = LabelEncoder()

    def remove_problematic_rows(self, X, threshold=1e15):
        initial_rows = X.shape[0]
        
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        X = X.dropna()
        X = X[~X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1)]
        X = X[~(X[numeric_columns].abs() > threshold).any(axis=1)]
        
        removed_rows = initial_rows - X.shape[0]
        self.logger.info(f"Removed {removed_rows} rows with NaN, infinite or extreme values")
        
        return X

    def simplify_attack_names(self, y):
        def map_attack(label):
            label = str(label).lower()
            
            if 'benign' in label:
                return 'Benign'
            elif any(attack in label for attack in ['dos', 'ddos', 'hulk', 'goldeneye', 'slowloris', 'slowhttptest']):
                return 'DoS Attack'
            elif any(attack in label for attack in ['web', 'xss', 'sql', 'injection', 'brute force']):
                return 'Web Attack'
            elif any(attack in label for attack in ['portscan', 'scanning', 'probe']):
                return 'Scanning And Probing'
            elif any(attack in label for attack in ['ftp', 'ssh', 'patator', 'password']):
                return 'Brute Force Attack'
            elif 'bot' in label:
                return 'Bot'
            elif 'infiltration' in label:
                return 'Infiltration'
            elif 'heartbleed' in label:
                return 'Heartbleed'
            else:
                return 'Other Attack'
        return pd.Series(y).apply(map_attack)

    @abstractmethod
    def engineer_features(self, X, y, dataset_name):
        pass

    def get_label_mapping(self):
        return dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))