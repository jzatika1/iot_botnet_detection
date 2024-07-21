from .feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler

class DLFeatureEngineer(FeatureEngineer):
    def __init__(self, config, logger, data_loader):
        super().__init__(config, logger, data_loader)
        self.scaler = StandardScaler()
        self.selected_features = None

    def process_features(self, X):
        self.logger.info("Processing DL features...")
        try:
            self.selected_features = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info(f"Shape after scaling: {X_scaled.shape}")
            
            # Save feature engineering metadata
            self.feature_engineering_metadata['scaler'] = self.scaler
            self.feature_engineering_metadata['selected_features'] = self.selected_features
            
            return X_scaled
        except Exception as e:
            self.logger.error(f"Error during feature processing: {str(e)}")
            raise