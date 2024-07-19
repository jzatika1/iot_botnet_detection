from .feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MLFeatureEngineer(FeatureEngineer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config['feature_engineering']['ml']['pca_components'])

    def process_features(self, X):
        self.logger.info("Processing ML features...")
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info(f"Shape after scaling: {X_scaled.shape}")
            
            X_pca = self.pca.fit_transform(X_scaled)
            self.logger.info(f"Shape after PCA: {X_pca.shape}")
            self.logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
            
            return X_pca
        except Exception as e:
            self.logger.error(f"Error during ML feature processing: {str(e)}")
            return None