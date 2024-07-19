import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.label_encoder = LabelEncoder()
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_mapping = None

    def engineer_features(self, X, y, dataset_name):
        self.logger.info(f"Starting feature engineering process for {dataset_name}")
        
        print(y)
        
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

    def remove_problematic_rows(self, X, y, threshold=1e15):
        initial_rows = X.shape[0]
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create a single mask for all problematic rows
        problematic_mask = (
            X.isna().any(axis=1) |  # NaN values
            X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1) |  # Infinite values
            (X[numeric_columns].abs() > threshold).any(axis=1)  # Extreme values
        )
        
        # Remove problematic rows
        X_clean = X[~problematic_mask].copy()
        y_clean = y[~problematic_mask].copy()
        
        removed_rows = initial_rows - X_clean.shape[0]
        nan_removed = X.isna().any(axis=1).sum()
        inf_removed = X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1).sum()
        extreme_removed = ((X[numeric_columns].abs() > threshold).any(axis=1) & ~X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1)).sum()
        
        self.logger.info(f"Removed {removed_rows} problematic rows:")
        self.logger.info(f"  - Rows with NaN values: {nan_removed}")
        self.logger.info(f"  - Rows with infinite values: {inf_removed}")
        self.logger.info(f"  - Rows with extreme values: {extreme_removed}")
        
        return X_clean, y_clean

    def preprocess_data(self, X, y):
        self.logger.info("Starting data preprocessing")
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        self.log_data_info(X)
        
        print(y)
        
        # Remove problematic rows
        X_clean, y_clean = self.remove_problematic_rows(X, y)
        
        # Convert to numeric
        X_numeric = self.convert_to_numeric(X_clean)
        
        self.logger.info(f"Preprocessing completed. Output shapes: X: {X_numeric.shape}, y: {y_clean.shape}")
        
        if X_numeric.shape[0] != y_clean.shape[0]:
            self.logger.error(f"Mismatch in number of rows: X has {X_numeric.shape[0]}, y has {y_clean.shape[0]}")
        
        return X_numeric, y_clean

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
        raise NotImplementedError("Subclasses must implement process_features method")

    def encode_labels(self, y):
        try:
            encoded_labels = self.label_encoder.fit_transform(y.astype(str))
            self.label_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
            self.logger.info(f"Label mapping: {self.label_mapping}")
            return encoded_labels
        except Exception as e:
            self.logger.error(f"Error during label encoding: {str(e)}")
            return None

    def log_data_info(self, X):
        self.logger.info(f"Shape of data: {X.shape}")
        nan_counts = X.isna().sum()
        self.logger.info(f"Total NaN values: {nan_counts.sum()}")
        self.logger.info("NaN values per column:")
        for column, count in nan_counts[nan_counts > 0].items():
            self.logger.info(f"  {column}: {count} ({count/len(X):.2%})")
        self.logger.info(f"Infinite values in X: {np.isinf(X.select_dtypes(include=np.number)).sum().sum()}")

    def save_processed_data(self, features, labels, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        features_path = os.path.join(output_dir, f'{prefix}_features.npy')
        labels_path = os.path.join(output_dir, f'{prefix}_labels.npy')
        metadata_path = os.path.join(output_dir, f'{prefix}_metadata.json')

        np.save(features_path, features)
        np.save(labels_path, labels)

        # Save metadata including number of classes and label mapping
        metadata = {
            "num_classes": len(np.unique(labels)),
            "feature_shape": features.shape,
            "label_mapping": self.label_mapping
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        self.logger.info(f"Processed data saved to {output_dir}")
        self.logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        self.logger.info(f"Number of classes: {metadata['num_classes']}")
        self.logger.info(f"Label mapping saved to metadata file")

    def load_processed_data(self, input_dir, prefix):
        features_path = os.path.join(input_dir, f'{prefix}_features.npy')
        labels_path = os.path.join(input_dir, f'{prefix}_labels.npy')
        metadata_path = os.path.join(input_dir, f'{prefix}_metadata.json')

        features = np.load(features_path)
        labels = np.load(labels_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.label_mapping = metadata.get("label_mapping")

        self.logger.info(f"Processed data loaded from {input_dir}")
        self.logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        self.logger.info(f"Number of classes: {metadata['num_classes']}")
        if self.label_mapping:
            self.logger.info(f"Label mapping loaded: {self.label_mapping}")
        else:
            self.logger.warning("No label mapping found in metadata")

        return features, labels, metadata