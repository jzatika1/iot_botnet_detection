import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from utils.logger import Logger

class FeatureEngineer:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or Logger.setup('FeatureEngineer')
        self.preprocessor = None
        self.svd = None
        self.label_encoder = LabelEncoder()

    def create_preprocessor(self, X):
        # Remove rows with any NaN values
        X = X.dropna(axis=0, how='any')
        
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        self.logger.info(f"Non-empty features: {len(X.columns)}")
        self.logger.info(f"Numeric features: {len(numeric_features)}")
        self.logger.info(f"Categorical features: {len(categorical_features)}")

        transformers = []

        if len(numeric_features) > 0:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))

        if len(categorical_features) > 0:
            # Transform categorical features using FeatureHasher
            def hash_categorical_features(df):
                hash_vector_size = 1000
                fh = FeatureHasher(n_features=hash_vector_size, input_type='string')
                hashed_features = fh.transform(df.astype(str).values)
                return hashed_features

            categorical_transformer = Pipeline(steps=[
                ('hasher', FunctionTransformer(hash_categorical_features, validate=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        if not transformers:
            raise ValueError("No non-empty features found in the dataset")

        self.preprocessor = ColumnTransformer(transformers=transformers)

        self.logger.info(f"Created preprocessor with {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")

    def remove_problematic_rows(self, X, threshold=1e15):
        initial_rows = X.shape[0]
        
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        X = X.dropna()
        X = X[~X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1)]
        X = X[~(X[numeric_columns].abs() > threshold).any(axis=1)]
        
        removed_rows = initial_rows - X.shape[0]
        self.logger.info(f"Removed {removed_rows} rows with NaN, infinite or extreme values")
        
        return X

    def engineer_features(self, X, y):
        self.logger.info("Starting feature engineering process")
    
        # Ensure X and y are pandas DataFrame/Series
        X = pd.DataFrame(X)
        y = pd.Series(y)
    
        # Simplify attack names
        y_simplified = simplify_attack_names(y)
        self.logger.info(f"Unique labels after simplification: {y_simplified.unique()}")
    
        # Store original index
        original_index = X.index
    
        X = self.remove_problematic_rows(X)
    
        # Update y to match X
        y_simplified = y_simplified.loc[X.index]
    
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_simplified)
    
        if self.preprocessor is None:
            self.create_preprocessor(X)
    
        # Ensure there are non-empty features
        if X.shape[1] == 0:
            raise ValueError("No non-empty features found after cleaning the data")
    
        # Preprocess data
        X_transformed = self.preprocessor.fit_transform(X)
    
        self.logger.info(f"Preprocessed data shape: {X_transformed.shape}")
        self.logger.info(f"Transformed data sample: {X_transformed[:5]}")
    
        if X_transformed.shape[1] == 0:
            raise ValueError("No features remaining after preprocessing")
    
        # Apply TruncatedSVD
        n_components = min(self.config['feature_engineering']['pca_components'], X_transformed.shape[1])
        if n_components == 0:
            raise ValueError("No components available for SVD")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_svd = self.svd.fit_transform(X_transformed)
    
        self.logger.info(f"Feature engineering completed. Output shape: {X_svd.shape}")
        self.logger.info(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
    
        return X_svd, y_encoded

    def get_label_mapping(self):
        return dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))

def get_feature_engineer(config, logger=None):
    return FeatureEngineer(config, logger)
    
def simplify_attack_names(y):
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