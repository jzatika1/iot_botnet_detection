import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self, config, logger, data_loader):
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        self.label_encoder = LabelEncoder()
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_mapping = None
        self.feature_engineering_metadata = {}

    def engineer_features(self, X, y, dataset_name):
        self.logger.info(f"Starting feature engineering process for {dataset_name}")
        
        X, y = self.preprocess_data(X, y)
        if X is None or y is None:
            self.logger.warning("Preprocessing returned None. Skipping further processing.")
            return None, None
    
        if X.isna().any().any():
            self.logger.info("NaN values detected. Performing imputation.")
            X_imputed = self.fast_impute(X)
            if X_imputed is None:
                self.logger.warning("Imputation failed. Returning original data.")
                return X, y
        else:
            self.logger.info("No NaN values detected. Skipping imputation.")
            X_imputed = X
    
        X_processed = self.process_features(X_imputed)
        if X_processed is None:
            self.logger.warning("Feature processing returned None. Returning imputed data.")
            return X_imputed, y
        
        y_encoded = self.encode_labels(y)
    
        if y_encoded is None:
            self.logger.warning("Label encoding returned None. Returning processed features and original labels.")
            return X_processed, y
    
        self.logger.info(f"Feature engineering completed. Output shape: {X_processed.shape}")
        return X_processed, y_encoded

    def preprocess_data(self, X, y):
        self.logger.info("Starting data preprocessing")
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Convert to numeric
        X_numeric = self.convert_to_numeric(X)
        
        if X_numeric.shape[0] != y.shape[0]:
            self.logger.error(f"Mismatch in number of rows: X has {X_numeric.shape[0]}, y has {y.shape[0]}")
        
        return X_numeric, y

    def convert_to_numeric(self, X):
        for column in X.select_dtypes(include=['object']):
            X.loc[:, column] = pd.factorize(X[column])[0]
        return X

    def fast_impute(self, X):
        self.logger.info("Imputing missing values...")
        try:
            numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns
            
            self.logger.info(f"Numeric columns: {len(numeric_columns)}, Categorical columns: {len(categorical_columns)}")
            
            if not numeric_columns.empty:
                X[numeric_columns] = self.numeric_imputer.fit_transform(X[numeric_columns])
            
            if not categorical_columns.empty:
                X[categorical_columns] = self.categorical_imputer.fit_transform(X[categorical_columns])
            
            self.logger.info(f"Shape after imputation: {X.shape}")
            return X
        except Exception as e:
            self.logger.error(f"Error during imputation: {str(e)}")
            return None

    def process_features(self, X):
        self.logger.info("Processing ML features...")
        try:
            original_columns = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            
            self.logger.info(f"PCA applied: reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} features, explaining {explained_variance:.2%} of variance")
            
            # Generate new feature names after PCA
            self.selected_features = [f'pca_component_{i}' for i in range(X_pca.shape[1])]
            
            # Save feature engineering metadata
            self.feature_engineering_metadata['scaler'] = self.scaler
            self.feature_engineering_metadata['pca'] = self.pca
            self.feature_engineering_metadata['original_columns'] = original_columns
            self.feature_engineering_metadata['selected_features'] = self.selected_features
            
            return X_pca
        except Exception as e:
            self.logger.error(f"Error during ML feature processing: {str(e)}")
            return None

    def encode_labels(self, y):
        try:
            encoded_labels = self.label_encoder.fit_transform(y.astype(str))
            self.label_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
            self.logger.info(f"Label mapping: {self.label_mapping}")
            return encoded_labels
        except Exception as e:
            self.logger.error(f"Error during label encoding: {str(e)}")
            return None

    def save_processed_data(self, features, labels, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        features_path = os.path.join(output_dir, f'{prefix}_features.npy')
        labels_path = os.path.join(output_dir, f'{prefix}_labels.npy')
        metadata_path = os.path.join(output_dir, f'{prefix}_metadata.json')
        fe_metadata_path = os.path.join(output_dir, f'{prefix}_fe_metadata.joblib')
    
        np.save(features_path, features)
        np.save(labels_path, labels)
    
        # Save metadata including number of classes, label mapping, and feature engineering metadata
        metadata = {
            "num_classes": len(np.unique(labels)),
            "feature_shape": features.shape,
            "label_mapping": self.label_mapping
        }
    
        # Add model-specific feature information
        if hasattr(self, 'pca_features'):
            metadata["original_features"] = self.original_features
            metadata["pca_features"] = self.pca_features
            metadata["pca_to_original_mapping"] = self.pca_to_original_mapping
        else:
            metadata["selected_features"] = self.selected_features
    
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
        # Save feature engineering metadata
        joblib.dump(self.feature_engineering_metadata, fe_metadata_path)
    
        self.logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        self.logger.info(f"Label mapping and feature engineering metadata saved")
        
    def load_processed_data(self, input_dir, prefix):
        features_path = os.path.join(input_dir, f'{prefix}_features.npy')
        labels_path = os.path.join(input_dir, f'{prefix}_labels.npy')
        metadata_path = os.path.join(input_dir, f'{prefix}_metadata.json')
        fe_metadata_path = os.path.join(input_dir, f'{prefix}_fe_metadata.joblib')
    
        features = np.load(features_path)
        labels = np.load(labels_path)
    
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
        self.label_mapping = metadata.get("label_mapping")
    
        # Load feature engineering metadata
        self.feature_engineering_metadata = joblib.load(fe_metadata_path)
    
        # Load PCA-specific information if it exists
        self.original_features = metadata.get("original_features")
        self.pca_features = metadata.get("pca_features")
        self.pca_to_original_mapping = metadata.get("pca_to_original_mapping")
    
        self.logger.info(f"Processed data loaded from {input_dir}. Features shape: {features.shape}, Labels shape: {labels.shape}")
        self.logger.info(f"Number of classes: {metadata['num_classes']}")
        self.logger.info("Label mapping and feature engineering metadata loaded")
    
        if self.pca_to_original_mapping:
            self.logger.info("PCA to original feature mapping loaded")
        
        return features, labels, metadata