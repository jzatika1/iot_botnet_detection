from .feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler

class DLFeatureEngineer(FeatureEngineer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.scaler = StandardScaler()

    def process_features(self, X):
        self.logger.info("Processing DL features...")
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info(f"Shape after scaling: {X_scaled.shape}")
            return X_scaled
        except Exception as e:
            self.logger.error(f"Error during DL feature processing: {str(e)}")
            return None