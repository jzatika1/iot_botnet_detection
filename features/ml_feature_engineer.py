from .feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class MLFeatureEngineer(FeatureEngineer):
    def __init__(self, config, logger, data_loader):
        super().__init__(config, logger, data_loader)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config['feature_engineering']['ml']['pca_components'])
        self.original_features = None
        self.pca_features = None
        self.pca_to_original_mapping = None

    def process_features(self, X):
        self.logger.info("Processing ML features...")
        try:
            self.original_features = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            self.logger.info(f"PCA applied: reduced from {X.shape[1]} to {X_pca.shape[1]} features, explaining {explained_variance:.4f} of variance")
            
            # Generate new feature names after PCA
            self.pca_features = [f'pca_component_{i}' for i in range(X_pca.shape[1])]
            
            # Create mapping between PCA components and original features
            self.pca_to_original_mapping = {}
            for i, pca_feature in enumerate(self.pca_features):
                component = self.pca.components_[i]
                top_features = [self.original_features[j] for j in np.argsort(np.abs(component))[-3:]]
                self.pca_to_original_mapping[pca_feature] = top_features
            
            # Save feature engineering metadata
            self.feature_engineering_metadata['scaler'] = self.scaler
            self.feature_engineering_metadata['pca'] = self.pca
            self.feature_engineering_metadata['original_features'] = self.original_features
            self.feature_engineering_metadata['pca_features'] = self.pca_features
            self.feature_engineering_metadata['pca_to_original_mapping'] = self.pca_to_original_mapping
            
            return X_pca
        except Exception as e:
            self.logger.error(f"Error during ML feature processing: {str(e)}")
            return None